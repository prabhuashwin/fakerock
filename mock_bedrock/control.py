"""Runtime control: modes.json, scenarios.json and the request log."""
import json
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

from mock_bedrock.packs.base import Call, Reply


def warn(msg):
    print(f"mock_bedrock: warning: {msg}", file=sys.stderr, flush=True)


class ScenarioError(ValueError):
    pass


def load_scenarios(path) -> list:
    """Load and validate a scenarios file; raises ScenarioError naming the offending scenario."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ScenarioError(f"cannot read scenarios file {path}: {e}")
    if not isinstance(data, dict) or not isinstance(data.get("scenarios", []), list):
        raise ScenarioError(f"scenarios file {path}: expected an object with a 'scenarios' list")
    out, seen = [], set()
    for n, sc in enumerate(data.get("scenarios", [])):
        if not isinstance(sc, dict):
            raise ScenarioError(f"scenario #{n}: must be an object")
        sid = sc.get("id")
        label = f"scenario {sid!r}" if sid is not None else f"scenario #{n}"
        if sid in seen:
            raise ScenarioError(f"{label}: duplicate id")
        seen.add(sid)
        m = sc.get("match") or {}
        if not isinstance(m, dict):
            raise ScenarioError(f"{label}: 'match' must be an object")
        for k in ("model", "prompt"):
            if k in m:
                try:
                    re.compile(m[k])
                except (re.error, TypeError) as e:
                    raise ScenarioError(f"{label}: match.{k} is not a valid regex: {e}")
        r = sc.get("reply") or {}
        if not isinstance(r, dict):
            raise ScenarioError(f"{label}: 'reply' must be an object")
        tu = r.get("tool_use")
        if tu is not None:
            if not isinstance(tu, dict) or not isinstance(tu.get("name"), str) or not tu["name"]:
                raise ScenarioError(f"{label}: reply.tool_use must be an object with a non-empty 'name'")
            if not isinstance(tu.get("input", {}), dict):
                raise ScenarioError(f"{label}: reply.tool_use.input must be an object")
        st = r.get("status", 200)
        if not isinstance(st, int) or isinstance(st, bool):
            raise ScenarioError(f"{label}: reply.status must be an integer")
        out.append(sc)
    return out


class Ctx:
    """Handed to packs: log_dir plus log(name, row) for extra jsonl files."""

    def __init__(self, log_dir, lock):
        self.log_dir = Path(log_dir)
        self._lock = lock
        self._warned = False

    def log(self, name, row):
        if not name.endswith(".jsonl"):
            name += ".jsonl"
        try:
            with self._lock, (self.log_dir / name).open("a", encoding="utf-8") as f:
                f.write(json.dumps({"at": time.time(), **row}) + "\n")
        except OSError as e:
            if not self._warned:
                self._warned = True
                warn(f"cannot write log {self.log_dir / name}: {e}")


class Control:
    def __init__(self, log_dir, modes_file=None, scenarios_file=None):
        requested = Path(log_dir)
        self.log_dir = requested
        try:
            requested.mkdir(parents=True, exist_ok=True)
            probe = requested / ".write-test"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as e:
            self.log_dir = Path(tempfile.mkdtemp(prefix="mock_bedrock_"))
            warn(f"log dir {requested} is not writable ({e}); logging to {self.log_dir} instead")
        self.modes_file = Path(modes_file) if modes_file else requested / "modes.json"
        self.lock = threading.Lock()
        self.ctx = Ctx(self.log_dir, self.lock)
        self._log_warned = False
        self._modes_good = {}
        self._modes_seen = None  # last raw content already parsed/reported
        self.scenarios = load_scenarios(scenarios_file) if scenarios_file else []

    def _modes(self) -> dict:
        try:
            raw = self.modes_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._modes_good, self._modes_seen = {}, None
            return self._modes_good
        except OSError as e:
            raw = f"<unreadable: {e}>"
            if raw != self._modes_seen:
                self._modes_seen = raw
                warn(f"cannot read {self.modes_file}: {e}; keeping previous modes")
            return self._modes_good
        if raw == self._modes_seen:
            return self._modes_good
        self._modes_seen = raw
        try:
            data = json.loads(raw or "{}")
            self._check_modes(data)
        except ValueError as e:
            warn(f"invalid {self.modes_file}: {e}; keeping previous modes")
            return self._modes_good
        self._modes_good = data
        return data

    @staticmethod
    def _check_modes(data):
        if not isinstance(data, dict):
            raise ValueError("top level must be an object")
        for fam, val in data.items():
            if fam == "_stall_seconds":
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise ValueError("_stall_seconds must be a number")
                continue
            if not isinstance(val, dict):
                raise ValueError(f"family {fam!r} must be an object")
            for needle, mode in val.items():
                if not isinstance(mode, str):
                    raise ValueError(f"mode for {fam!r}/{needle!r} must be a string")

    def mode_for(self, pack, call: Call, family: str) -> str:
        fam = self._modes().get(family) or {}
        key = pack.mode_key(call, family).lower()
        for needle, mode in fam.items():
            if needle != "_default" and needle.lower() in key:
                return mode
        return fam.get("_default") or pack.default_mode(family)

    def stall_seconds(self) -> float:
        return float(self._modes().get("_stall_seconds", 60))

    def scenario_for(self, call: Call):
        for sc in self.scenarios:
            m = sc.get("match") or {}
            if "model" in m and not re.search(m["model"], call.model):
                continue
            if "prompt" in m and not re.search(m["prompt"], call.prompt):
                continue
            r = sc.get("reply") or {}
            tu = r.get("tool_use")
            if tu is not None:
                tu = {"id": tu.get("id", "tooluse_scenario"), "name": tu["name"], "input": tu.get("input", {})}
            status = int(r.get("status", 200))
            return Reply(text=r.get("text", ""), tool_use=tu,
                         stop_reason=r.get("stop_reason", "tool_use" if tu else "end_turn"),
                         status=status, error_type=r.get("error_type"),
                         message=r.get("message", r.get("text", "") if status >= 400 else ""))
        return None

    def log_request(self, row: dict):
        try:
            with self.lock, (self.log_dir / "requests.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:
            if not self._log_warned:
                self._log_warned = True
                warn(f"cannot write request log in {self.log_dir}: {e}")
