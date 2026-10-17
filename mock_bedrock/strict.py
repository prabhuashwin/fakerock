"""Opt-in strict request validation (--strict): reject requests real Bedrock would reject.

check_converse(body) and check_invoke(model, body) return a ValidationException message, or None if valid.
"""
import re

from mock_bedrock.embeddings import is_embedding_model

TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
INVOKE_KEYS = ("anthropic_version", "anthropic_beta", "max_tokens", "messages", "system", "temperature", "top_p",
               "top_k", "stop_sequences", "tools", "tool_choice", "metadata", "thinking", "output_config",
               "output_format", "context_management", "container", "mcp_servers")


def is_anthropic_invoke(model, body):
    return not is_embedding_model(model) and ("anthropic_version" in body or "anthropic." in model)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _blocks(m, conv):
    c = m.get("content")
    return [b for b in c if isinstance(b, dict)] if isinstance(c, list) else []


def _uses(m, conv):
    if conv:
        ids = [b["toolUse"].get("toolUseId") for b in _blocks(m, conv) if isinstance(b.get("toolUse"), dict)]
    else:
        ids = [b.get("id") for b in _blocks(m, conv) if b.get("type") == "tool_use"]
    return [i for i in ids if i is not None]


def _results(m, conv):
    """(block index, tool use id) for each tool result block."""
    out = []
    c = m.get("content")
    for j, b in enumerate(c if isinstance(c, list) else []):
        if not isinstance(b, dict):
            continue
        if conv and isinstance(b.get("toolResult"), dict):
            out.append((j, b["toolResult"].get("toolUseId")))
        elif not conv and b.get("type") == "tool_result":
            out.append((j, b.get("tool_use_id")))
    return out


def _malformed(c, i, conv):
    """Explicit rejection of non-string text / tool ids (checked before any id matching or hashing)."""
    for j, b in enumerate(c if isinstance(c, list) else []):
        if not isinstance(b, dict):
            continue
        where = f"messages.{i}.content.{j}"
        if "text" in b and (conv or b.get("type") == "text") and not isinstance(b["text"], str):
            return f"{where}.text must be a string"
        if conv:
            for key in ("toolUse", "toolResult"):
                if isinstance(b.get(key), dict) and "toolUseId" in b[key] and not isinstance(b[key]["toolUseId"], str):
                    return f"{where}.{key}.toolUseId must be a string"
            for k, r in enumerate(b["toolResult"].get("content") or [] if isinstance(b.get("toolResult"), dict)
                                  and isinstance(b["toolResult"].get("content"), list) else []):
                if isinstance(r, dict) and "text" in r and not isinstance(r["text"], str):
                    return f"{where}.toolResult.content.{k}.text must be a string"
        else:
            key = {"tool_use": "id", "tool_result": "tool_use_id"}.get(b.get("type"))
            if key and key in b and not isinstance(b[key], str):
                return f"{where}.{key} must be a string"
    return None


def _messages(msgs, conv):
    """Structure checks shared by converse and invoke messages; returns (error, saw_tool_blocks)."""
    first = ("A conversation must start with a user message. Try again with a conversation that starts with a "
             "user message." if conv else 'messages: first message must use the "user" role')
    if not isinstance(msgs, list) or not msgs:
        return (first if conv else "messages: at least one message is required"), False
    tools = False
    for i, m in enumerate(msgs):
        if not isinstance(m, dict):
            return f"messages.{i} must be an object", tools
        role = m.get("role")
        if i == 0 and role != "user" and role in ("user", "assistant"):
            return first, tools
        if role not in ("user", "assistant"):
            return (f"messages.{i}.role must be one of [user, assistant]" if conv
                    else f"messages.{i}.role: Input should be 'user' or 'assistant'"), tools
        if i and msgs[i - 1].get("role") == role:
            return ("A conversation must alternate between user and assistant roles. Make sure the conversation "
                    "alternates between user and assistant roles and try again." if conv else
                    f'messages: roles must alternate between "user" and "assistant", but found multiple "{role}" '
                    "roles in a row"), tools
        c = m.get("content")
        bad = _malformed(c, i, conv)
        if bad:
            return bad, tools
        if isinstance(c, str) and not conv:
            if not c.strip():
                return "messages: text content blocks must be non-empty", tools
        elif not isinstance(c, list) or not c:
            return (f"The content field in the Message object at messages.{i} is empty. Add a ContentBlock object "
                    "to the content field and try again." if conv else
                    f"messages.{i}: content must be a non-empty string or list of blocks"), tools
        for j, b in enumerate(c if isinstance(c, list) else []):
            text = b.get("text") if isinstance(b, dict) and (conv or b.get("type") == "text") else None
            if isinstance(text, str) and not text.strip():
                return (f"The text field in the ContentBlock object at messages.{i}.content.{j} is blank. Add text "
                        "to the text field, and try again." if conv else
                        "messages: text content blocks must be non-empty"), tools
        prev_uses = _uses(msgs[i - 1], conv) if i and msgs[i - 1].get("role") == "assistant" else []
        results = _results(m, conv)
        tools = tools or bool(results or _uses(m, conv))
        if role == "user" and (results or prev_uses):
            if conv and len(results) > len(prev_uses):
                return f"The number of toolResult blocks at messages.{i}.content exceeds the number of toolUse blocks of previous turn.", tools
            for j, rid in results:
                if rid not in prev_uses:
                    return (f"The toolUseId {rid} in messages.{i}.content.{j} doesn't match any toolUse block in the "
                            "previous turn." if conv else
                            f"messages.{i}.content.{j}: unexpected `tool_use_id` found in `tool_result` blocks: "
                            f"{rid}. Each `tool_result` block must have a corresponding `tool_use` block in the "
                            "previous message."), tools
            missing = [u for u in prev_uses if u not in {r for _, r in results}]
            if missing:
                return (f"Expected toolResult blocks at messages.{i}.content for the following Ids: "
                        f"{', '.join(map(str, missing))}" if conv else
                        f"messages.{i}: `tool_use` ids were found without `tool_result` blocks immediately after: "
                        f"{', '.join(map(str, missing))}. Each `tool_use` block must have a corresponding "
                        "`tool_result` block in the next message."), tools
    return None, tools


def _tool_names(tools, conv):
    names = []
    for k, t in enumerate(tools):
        if isinstance(t, dict) and "cachePoint" in t and "toolSpec" not in t:
            continue
        spec = t.get("toolSpec") if isinstance(t, dict) else None
        name = spec.get("name") if isinstance(spec, dict) else None
        if not isinstance(name, str) or not TOOL_NAME_RE.match(name):
            return None, (f"The tool name {name!r} at toolConfig.tools.{k} is invalid. Tool names must match "
                          "[a-zA-Z0-9_-]{1,64}.")
        if name in names:
            return None, f"Tool names must be unique: {name}"
        schema = spec.get("inputSchema")
        if not isinstance(schema, dict) or not isinstance(schema.get("json"), dict):
            return None, f"toolConfig.tools.{k}.toolSpec.inputSchema.json must be an object"
        names.append(name)
    return names, None


def check_converse(body):
    err, tools = _messages(body.get("messages"), True)
    if err:
        return err
    cfg = body.get("inferenceConfig")
    if cfg is not None:
        if not isinstance(cfg, dict):
            return "inferenceConfig must be an object"
        v = cfg.get("maxTokens")
        if "maxTokens" in cfg and (not _int(v) or v < 1):
            return "The value at inferenceConfig.maxTokens must be an integer greater than or equal to 1."
        for k in ("temperature", "topP"):
            if k in cfg and (not _num(cfg[k]) or not 0 <= cfg[k] <= 1):
                return f"The value at inferenceConfig.{k} must be between 0 and 1."
        s = cfg.get("stopSequences")
        if "stopSequences" in cfg and (not isinstance(s, list) or not all(isinstance(x, str) for x in s)):
            return "The value at inferenceConfig.stopSequences must be a list of strings."
    tc = body.get("toolConfig")
    if tc is None:
        if tools:
            return "The toolConfig field must be defined when using toolUse and toolResult content blocks."
        return None
    if not isinstance(tc, dict) or not isinstance(tc.get("tools"), list) or not tc["tools"]:
        return "toolConfig.tools must contain at least one tool"
    return _tool_names(tc["tools"], True)[1]


def check_invoke(model, body):
    def bad(m):
        return f"Malformed input request: {m}, please reformat your input and try again."
    for k in ("anthropic_version", "max_tokens", "messages"):
        if k not in body:
            return bad(f"#: required key [{k}] not found")
    for k in body:
        if k not in INVOKE_KEYS:
            return bad(f"#: extraneous key [{k}] is not permitted")
    if body["anthropic_version"] != "bedrock-2023-05-31":
        return bad(f"#/anthropic_version: {body['anthropic_version']!r} is not a valid enum value")
    v = body["max_tokens"]
    if not _int(v) or v < 1:
        return bad(f"#/max_tokens: {v} is not greater or equal to 1")
    for k in ("temperature", "top_p"):
        if k in body:
            v = body[k]
            if not _num(v):
                return bad(f"#/{k}: expected type: Number, found: {type(v).__name__}")
            if v < 0:
                return bad(f"#/{k}: {v} is not greater or equal to 0")
            if v > 1:
                return bad(f"#/{k}: {v} is not less or equal to 1.0")
    if "top_k" in body and (not _int(body["top_k"]) or body["top_k"] < 0):
        return bad(f"#/top_k: {body['top_k']} is not greater or equal to 0")
    err, _ = _messages(body["messages"], False)
    return err
