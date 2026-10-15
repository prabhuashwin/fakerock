"""Runtime control: modes.json, scenarios.json and the request log."""
import collections
import json
import math
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
    return validate_scenarios(data, f"scenarios file {path}")


# mode -> (HTTP status, error type, message); statuses match botocore's bedrock-runtime model.
ERROR_MODES = {
    "throttle": (429, "ThrottlingException", "Too many requests (mock)"),
    "unavailable": (503, "ServiceUnavailableException", "Service unavailable (mock)"),
    "error": (500, "InternalServerException", "Internal server error (mock)"),
    "validation": (400, "ValidationException", "Validation error (mock)"),
    "model_timeout": (408, "ModelTimeoutException", "Model timed out (mock)"),
    "model_not_ready": (429, "ModelNotReadyException", "Model not ready (mock)"),
    "access_denied": (403, "AccessDeniedException", "Access denied (mock)"),
    "not_found": (404, "ResourceNotFoundException", "Resource not found (mock)"),
    "quota_exceeded": (400, "ServiceQuotaExceededException", "Service quota exceeded (mock)"),
    "model_error": (424, "ModelErrorException", "Model error (mock)"),
}
COMMON_MODES = ("answer", "stall", "truncate", *ERROR_MODES, "stream_error", "drop")


def _validate_reply(r, label):
    if not isinstance(r, dict):
        raise ScenarioError(f"{label} must be an object")
    mode = r.get("mode")
    if mode is not None and mode not in COMMON_MODES:
        raise ScenarioError(f"{label}.mode must be one of {', '.join(COMMON_MODES)}")
    tu = r.get("tool_use")
    if tu is not None:
        if not isinstance(tu, dict) or not isinstance(tu.get("name"), str) or not tu["name"]:
            raise ScenarioError(f"{label}.tool_use must be an object with a non-empty 'name'")
        if not isinstance(tu.get("input", {}), dict):
            raise ScenarioError(f"{label}.tool_use.input must be an object")
    st = r.get("status", 200)
    if not isinstance(st, int) or isinstance(st, bool):
        raise ScenarioError(f"{label}.status must be an integer")


def validate_scenarios(data, source="scenarios") -> list:
    """Validate parsed scenarios (object with a 'scenarios' list); returns the list."""
    if not isinstance(data, dict) or not isinstance(data.get("scenarios", []), list):
        raise ScenarioError(f"{source}: expected an object with a 'scenarios' list")
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
        if "reply" in sc and "replies" in sc:
            raise ScenarioError(f"{label}: use either 'reply' or 'replies', not both")
        if "times" in sc:
            t = sc["times"]
            if not isinstance(t, int) or isinstance(t, bool) or t < 1:
                raise ScenarioError(f"{label}: 'times' must be a positive integer")
        if "replies" in sc:
            rs = sc["replies"]
            if not isinstance(rs, list) or not rs:
                raise ScenarioError(f"{label}: 'replies' must be a non-empty list of reply objects")
            for i, r in enumerate(rs):
                _validate_reply(r, f"{label}: replies[{i}]")
        else:
            _validate_reply(sc.get("reply") or {}, f"{label}: reply")
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
    def __init__(self, log_dir, modes_file=None, scenarios_file=None, *, modes=None, scenarios=None):
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
        self._modes_lock = threading.Lock()  # guards modes-file reload state; separate from self.lock
        self.ctx = Ctx(self.log_dir, self.lock)
        self._log_warned = False
        self._modes_good = {}
        self._modes_seen = None  # last raw content already parsed/reported
        if scenarios is not None:
            self._startup_scenarios = validate_scenarios(
                {"scenarios": scenarios} if isinstance(scenarios, list) else scenarios)
        else:
            self._startup_scenarios = load_scenarios(scenarios_file) if scenarios_file else []
        self.scenarios = self._startup_scenarios
        self._modes_fixed = None
        if modes is not None:
            self._check_modes(modes)
            self._modes_fixed = modes
        self._modes_override = None  # set via set_modes(); wins over file/init modes until reset()
        self._requests = collections.deque(maxlen=10000)
        self._seq = 0
        # Callables run by reset(); later steps (per-scenario counters) register here.
        self._reset_hooks = [self._counts_clear]
        self._counts = {}  # scenario list index -> times selected

    def _counts_clear(self):
        self._counts.clear()  # called with self.lock held

    def requests(self) -> list:
        with self.lock:
            return list(self._requests)

    def set_scenarios(self, data) -> int:
        if isinstance(data, list):
            data = {"scenarios": data}
        new = validate_scenarios(data, "scenarios")
        with self.lock:
            self.scenarios = new
            self._counts.clear()
        return len(new)

    def set_modes(self, data):
        try:
            self._check_modes(data)
        except ValueError as e:
            raise ValueError(f"invalid modes: {e}")
        with self.lock:
            self._modes_override = data

    def reset(self):
        with self.lock:
            self._requests.clear()
            self._seq = 0
            self._modes_override = None
            self.scenarios = self._startup_scenarios
            for hook in self._reset_hooks:
                hook()

    def _modes(self) -> dict:
        override = self._modes_override
        if override is not None:
            return override
        if self._modes_fixed is not None:
            return self._modes_fixed
        with self._modes_lock:
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
            if fam in ("_stall_seconds", "_latency_seconds", "_chunk_delay_seconds"):
                if (isinstance(val, bool) or not isinstance(val, (int, float))
                        or not (val >= 0 and math.isfinite(val))):
                    raise ValueError(f"{fam} must be a finite number >= 0")
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

    def latency_seconds(self) -> float:
        return float(self._modes().get("_latency_seconds", 0))

    def chunk_delay_seconds(self) -> float:
        return float(self._modes().get("_chunk_delay_seconds", 0))

    def select_scenario(self, call: Call):
        """First matching scenario with 'times' left; counts the match atomically.
        Returns (scenario_id, reply_spec_dict) or None."""
        with self.lock:
            for idx, sc in enumerate(self.scenarios):
                m = sc.get("match") or {}
                if "model" in m and not re.search(m["model"], call.model):
                    continue
                if "prompt" in m and not re.search(m["prompt"], call.prompt):
                    continue
                n = self._counts.get(idx, 0)
                times = sc.get("times")
                if times is not None and n >= times:
                    continue
                self._counts[idx] = n + 1
                if "replies" in sc:
                    spec = sc["replies"][min(n, len(sc["replies"]) - 1)]
                else:
                    spec = sc.get("reply") or {}
                return sc.get("id"), spec
        return None

    @staticmethod
    def reply_from_spec(r: dict) -> Reply:
        tu = r.get("tool_use")
        if tu is not None:
            tu = {"id": tu.get("id", "tooluse_scenario"), "name": tu["name"], "input": tu.get("input", {})}
        status = int(r.get("status", 200))
        return Reply(text=r.get("text", ""), tool_use=tu,
                     stop_reason=r.get("stop_reason", "tool_use" if tu else "end_turn"),
                     status=status, error_type=r.get("error_type"),
                     message=r.get("message", r.get("text", "") if status >= 400 else ""))

    def scenario_for(self, call: Call):
        sel = self.select_scenario(call)
        return self.reply_from_spec(sel[1]) if sel else None

    def log_request(self, row: dict):
        with self.lock:
            self._seq += 1
            row = {"seq": self._seq, **row}
            self._requests.append(row)
        try:
            with self.lock, (self.log_dir / "requests.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError as e:
            if not self._log_warned:
                self._log_warned = True
                warn(f"cannot write request log in {self.log_dir}: {e}")
