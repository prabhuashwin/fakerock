"""Anthropic-messages InvokeModel: pure parse/build functions."""
import base64
import json
import uuid

from mock_bedrock import eventstream
from mock_bedrock.packs.base import Call, Reply


def split_chunks(text: str, size: int = 20) -> list:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in (content or []) if isinstance(b, dict)]


def _int_field(value, name, default):
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def parse_invoke(model: str, body: dict, stream: bool = False) -> Call:
    """Raises ValueError on a malformed body (the server turns it into a 400)."""
    if not isinstance(body.get("messages"), list):
        raise ValueError("messages must be a list")
    if not all(isinstance(m, dict) for m in body["messages"]):
        raise ValueError("each message must be an object")
    max_tokens = _int_field(body.get("max_tokens"), "max_tokens", 1024)
    system = body.get("system", "")
    if isinstance(system, list):
        system = "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
    messages = []
    for m in body["messages"]:
        texts, results, uses = [], [], []
        for b in _blocks(m.get("content")):
            t = b.get("type")
            if t == "text":
                texts.append(b.get("text", ""))
            elif t == "tool_use":
                uses.append({"id": b.get("id"), "name": b.get("name"), "input": b.get("input", {})})
            elif t == "tool_result":
                results.append(b)
        messages.append({"role": m.get("role", "user"), "text": "\n".join(texts),
                         "tool_results": results, "tool_uses": uses})
    tools = [{"name": t.get("name"), "description": t.get("description", ""),
              "input_schema": t.get("input_schema") or {}} for t in body.get("tools", []) or []]
    return Call(api="invoke", model=model, system=system or "", messages=messages, tools=tools,
                max_tokens=max_tokens, stream=stream, raw=body)


def invoke_stream_events(call: Call, reply: Reply, stream_error: bool = False) -> list:
    """Encoded InvokeModelWithResponseStream events (Anthropic stream events in chunk.bytes)."""
    def chunk(obj):
        raw = base64.b64encode(json.dumps(obj).encode("utf-8")).decode("ascii")
        return eventstream.event("chunk", {"bytes": raw})

    in_tok, out_tok = max(1, len(call.prompt) // 4), reply.tokens(call.max_tokens)
    ev = [chunk({"type": "message_start", "message": {
        "id": "msg_mock_" + uuid.uuid4().hex[:16], "type": "message", "role": "assistant",
        "model": call.model, "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": 1}}})]
    if reply.tool_use:
        tu = reply.tool_use
        ev.append(chunk({"type": "content_block_start", "index": 0, "content_block": {
            "type": "tool_use", "id": tu["id"], "name": tu["name"], "input": {}}}))
        deltas = [{"type": "input_json_delta", "partial_json": json.dumps(tu.get("input", {}))}]
    else:
        ev.append(chunk({"type": "content_block_start", "index": 0,
                         "content_block": {"type": "text", "text": ""}}))
        deltas = [{"type": "text_delta", "text": c} for c in split_chunks(reply.text)]
    if stream_error:
        ev.append(chunk({"type": "content_block_delta", "index": 0, "delta": deltas[0]}))
        ev.append(eventstream.exception("modelStreamErrorException", "Mock stream error"))
        return ev
    for d in deltas:
        ev.append(chunk({"type": "content_block_delta", "index": 0, "delta": d}))
    ev.append(chunk({"type": "content_block_stop", "index": 0}))
    ev.append(chunk({"type": "message_delta", "delta": {"stop_reason": reply.stop_reason, "stop_sequence": None},
                     "usage": {"output_tokens": out_tok}}))
    ev.append(chunk({"type": "message_stop", "amazon-bedrock-invocationMetrics": {
        "inputTokenCount": in_tok, "outputTokenCount": out_tok,
        "invocationLatency": 5, "firstByteLatency": 2}}))
    return ev


def build_invoke_response(call: Call, reply: Reply) -> dict:
    if reply.tool_use:
        tu = reply.tool_use
        content = [{"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": tu.get("input", {})}]
    else:
        content = [{"type": "text", "text": reply.text}]
    return {"id": "msg_mock_" + uuid.uuid4().hex[:16], "type": "message", "role": "assistant",
            "model": call.model, "content": content, "stop_reason": reply.stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": max(1, len(call.prompt) // 4),
                      "output_tokens": reply.tokens(call.max_tokens)}}
