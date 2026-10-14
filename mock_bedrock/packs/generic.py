"""Generic pack: echo replies, and a tool call when tools are offered."""
import itertools

from mock_bedrock.packs.base import Call, Pack, Reply


def build_input(schema: dict) -> dict:
    schema = schema or {}
    props = schema.get("properties") or {}
    out = {}
    for name in schema.get("required") or []:
        out[name] = sample_value(props.get(name) or {})
    return out


def sample_value(prop: dict):
    if prop.get("enum"):
        return prop["enum"][0]
    t = prop.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t == "string":
        return "mock"
    if t in ("integer", "number"):
        return 0
    if t == "boolean":
        return False
    if t == "array":
        return []
    if t == "object":
        return {}
    return "mock"


class GenericPack(Pack):
    name = "generic"

    def __init__(self, options=None):
        super().__init__(options)
        self._counter = itertools.count(1)

    def family(self, call: Call) -> str:
        return "tools" if call.tools else "chat"

    def reply(self, call: Call, family: str, mode: str, ctx) -> Reply:
        last = call.messages[-1] if call.messages else {}
        if call.tools and not last.get("tool_results"):
            tool = call.tools[0]
            return Reply(
                tool_use={"id": f"tooluse_mock_{next(self._counter)}", "name": tool["name"],
                          "input": build_input(tool.get("input_schema"))},
                stop_reason="tool_use")
        return Reply(text="Mock reply to: " + call.last_user_text[:200])


PACK = GenericPack
