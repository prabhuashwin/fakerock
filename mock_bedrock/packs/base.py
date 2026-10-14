"""Pack contract: Call, Reply, Pack and load_pack."""
import importlib
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Call:
    api: str                      # "invoke" | "converse"
    model: str
    system: str = ""
    # {"role", "text", "tool_results": list, "tool_uses": list}
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=list)   # {"name", "description", "input_schema"}
    max_tokens: int = 1024
    stream: bool = False
    raw: dict = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return self.system + "\n" + "\n".join(m.get("text", "") for m in self.messages)

    @property
    def last_user_text(self) -> str:
        for m in reversed(self.messages):
            if m.get("role") == "user" and m.get("text"):
                return m["text"]
        for m in reversed(self.messages):
            if m.get("role") == "user":
                return m.get("text", "")
        return ""


@dataclass
class Reply:
    text: str = ""
    tool_use: Optional[dict] = None       # {"id", "name", "input"}
    stop_reason: str = "end_turn"
    output_tokens: Optional[int] = None   # None -> max(1, len(text)//4)
    status: int = 200
    error_type: Optional[str] = None
    raw_body: Optional[bytes] = None      # sent verbatim with status
    message: str = ""                     # error message

    def tokens(self, cap: Optional[int] = None) -> int:
        n = self.output_tokens if self.output_tokens is not None else max(1, len(self.text) // 4)
        return min(n, cap) if cap is not None and cap > 0 else n


class Pack:
    name = "base"

    def __init__(self, options: Optional[dict] = None):
        self.options = options or {}

    def family(self, call: Call) -> str:
        return "chat"

    def default_mode(self, family: str) -> str:
        return "answer"

    def mode_key(self, call: Call, family: str) -> str:
        return call.last_user_text

    def reply(self, call: Call, family: str, mode: str, ctx: Any) -> Reply:
        raise NotImplementedError


def load_pack(spec: str, options: Optional[dict] = None) -> Pack:
    options = options or {}
    if spec == "generic":
        from mock_bedrock.packs.generic import PACK
        return PACK(options)
    path = Path(spec)
    if spec.endswith(".py"):
        if not path.is_file():
            raise RuntimeError(f"pack file not found: {spec}")
        mspec = importlib.util.spec_from_file_location("mock_bedrock_user_pack", str(path))
        mod = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(mod)
        if not hasattr(mod, "PACK"):
            raise RuntimeError(f"pack file {spec} does not define PACK")
        return mod.PACK(options)
    raise RuntimeError(f"unknown pack {spec!r}: use 'generic' or a path to a .py file defining PACK")
