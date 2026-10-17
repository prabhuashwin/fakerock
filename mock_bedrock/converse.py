"""Bedrock Converse / ConverseStream: pure parse/build functions."""
import json

from mock_bedrock import eventstream
from mock_bedrock.packs.base import Call, Reply


def split_chunks(text: str, size: int = 20) -> list:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _opt_str(value, path):
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    return value


def _result_content(items):
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        if "text" in it:
            if not isinstance(it["text"], str):
                raise ValueError("toolResult content text must be a string")
            out.append({"type": "text", "text": it["text"]})
        elif "json" in it:
            out.append({"type": "text", "text": json.dumps(it["json"])})
    return out


def parse_converse(model: str, body: dict, stream: bool = False) -> Call:
    """Raises ValueError on a malformed body (the server turns it into a 400)."""
    if not isinstance(body.get("messages"), list):
        raise ValueError("messages must be a list")
    if not all(isinstance(m, dict) for m in body["messages"]):
        raise ValueError("each message must be an object")
    cfg = body.get("inferenceConfig")
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("inferenceConfig must be an object")
    max_tokens = cfg.get("maxTokens", 1024)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("inferenceConfig.maxTokens must be an integer")
    sys_blocks = body.get("system", []) or []
    if not isinstance(sys_blocks, list):
        raise ValueError("system must be a list")
    system_texts = []
    for k, b in enumerate(sys_blocks):
        if isinstance(b, dict) and "text" in b:
            if not isinstance(b["text"], str):
                raise ValueError(f"system.{k}.text must be a string")
            system_texts.append(b["text"])
    system = "\n".join(system_texts)
    messages = []
    for i, m in enumerate(body["messages"]):
        texts, results, uses = [], [], []
        content = m.get("content", []) or []
        if not isinstance(content, list):
            raise ValueError(f"messages.{i}.content must be a list")
        for j, b in enumerate(content):
            if not isinstance(b, dict):
                continue
            where = f"messages.{i}.content.{j}"
            if "text" in b:
                if not isinstance(b["text"], str):
                    raise ValueError(f"{where}.text must be a string")
                texts.append(b["text"])
            elif isinstance(b.get("toolUse"), dict):
                tu = b["toolUse"]
                _opt_str(tu.get("toolUseId"), f"{where}.toolUse.toolUseId")
                uses.append({"id": tu.get("toolUseId"), "name": tu.get("name"), "input": tu.get("input", {})})
            elif isinstance(b.get("toolResult"), dict):
                tr = b["toolResult"]
                _opt_str(tr.get("toolUseId"), f"{where}.toolResult.toolUseId")
                results.append({"type": "tool_result", "tool_use_id": tr.get("toolUseId"),
                                "content": _result_content(tr.get("content")),
                                "is_error": tr.get("status") == "error"})
        messages.append({"role": m.get("role", "user"), "text": "\n".join(texts),
                         "tool_results": results, "tool_uses": uses})
    tools = []
    tool_config = body.get("toolConfig") or {}
    if not isinstance(tool_config, dict):
        raise ValueError("toolConfig must be an object")
    raw_tools = tool_config.get("tools") or []
    if not isinstance(raw_tools, list):
        raise ValueError("toolConfig.tools must be a list")
    for t in raw_tools:
        spec = t.get("toolSpec") if isinstance(t, dict) else None
        if not isinstance(spec, dict):
            continue
        tools.append({"name": spec.get("name"), "description": spec.get("description", ""),
                      "input_schema": (spec.get("inputSchema") or {}).get("json") or {}})
    return Call(api="converse", model=model, system=system, messages=messages, tools=tools,
                max_tokens=max_tokens, stream=stream, raw=body)


def _usage(call: Call, reply: Reply) -> dict:
    i, o = max(1, len(call.prompt) // 4), reply.tokens(call.max_tokens)
    return {"inputTokens": i, "outputTokens": o, "totalTokens": i + o}


def build_converse_response(call: Call, reply: Reply) -> dict:
    if reply.tool_use:
        tu = reply.tool_use
        content = [{"toolUse": {"toolUseId": tu["id"], "name": tu["name"], "input": tu.get("input", {})}}]
    else:
        content = [{"text": reply.text}]
    return {"output": {"message": {"role": "assistant", "content": content}},
            "stopReason": reply.stop_reason, "usage": _usage(call, reply),
            "metrics": {"latencyMs": 5}}


def converse_stream_events(call: Call, reply: Reply, stream_error=None, *, error_after: int = 1) -> list:
    """Encoded events. stream_error: None/False, True (modelStreamErrorException) or an exception type;
    when set, messageStart, error_after deltas, then that exception event."""
    ev = [eventstream.event("messageStart", {"role": "assistant"})]
    if reply.tool_use:
        tu = reply.tool_use
        ev.append(eventstream.event("contentBlockStart", {"contentBlockIndex": 0, "start": {
            "toolUse": {"toolUseId": tu["id"], "name": tu["name"]}}}))
        deltas = [{"toolUse": {"input": json.dumps(tu.get("input", {}))}}]
    else:
        deltas = [{"text": c} for c in split_chunks(reply.text)]
    if stream_error:
        for d in deltas[:error_after]:
            ev.append(eventstream.event("contentBlockDelta", {"contentBlockIndex": 0, "delta": d}))
        ev.append(eventstream.exception(
            "modelStreamErrorException" if stream_error is True else stream_error, "Mock stream error"))
        return ev
    for d in deltas:
        ev.append(eventstream.event("contentBlockDelta", {"contentBlockIndex": 0, "delta": d}))
    ev.append(eventstream.event("contentBlockStop", {"contentBlockIndex": 0}))
    ev.append(eventstream.event("messageStop", {"stopReason": reply.stop_reason}))
    ev.append(eventstream.event("metadata", {"usage": _usage(call, reply), "metrics": {"latencyMs": 5}}))
    return ev
