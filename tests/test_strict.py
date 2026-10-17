"""Strict mode: requests real Bedrock would reject get a 400 ValidationException; valid ones pass."""
import copy
import json
import urllib.error
import urllib.request

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.config import Config
from botocore.exceptions import ClientError

from mock_bedrock.testing import MockBedrock

CLAUDE = "anthropic.claude-test"
TITAN = "amazon.titan-embed-text-v2:0"
CFG = Config(retries={"total_max_attempts": 1, "mode": "standard"}, read_timeout=5, connect_timeout=5)
U = {"role": "user", "content": [{"text": "hi"}]}
A = {"role": "assistant", "content": [{"text": "hello"}]}
TOOL = {"toolSpec": {"name": "get_x", "inputSchema": {"json": {"type": "object"}}}}
TC = {"tools": [TOOL]}
USE = {"toolUse": {"toolUseId": "t1", "name": "get_x", "input": {}}}
RES = {"toolResult": {"toolUseId": "t1", "content": [{"text": "r"}]}}
START = "A conversation must start with a user message"
ALT = "A conversation must alternate between user and assistant roles"
EMPTY = "The content field in the Message object at messages.0 is empty"
IV = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture(scope="module")
def mock():
    with MockBedrock(strict=True) as m:
        yield m


@pytest.fixture(scope="module")
def br(mock):
    return mock.client(config=CFG)


def post(mock, path, body, model=CLAUDE):
    """Raw POST; returns (status, error type header, parsed body)."""
    req = urllib.request.Request(f"{mock.endpoint_url}/model/{model}/{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, None, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("x-amzn-ErrorType"), json.loads(e.read() or b"{}")


def raw_post(mock, path, body, model):
    """Raw POST; returns (status, body text) without parsing."""
    req = urllib.request.Request(f"{mock.endpoint_url}/model/{model}/{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode("latin-1")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("latin-1")


def invoke_body(**kw):
    return {**copy.deepcopy(IV), **kw}


def expect_400(mock, path, body, substr):
    status, etype, data = post(mock, path, body)
    assert status == 400, data
    assert (etype or "").startswith("ValidationException") or "ValidationException" in json.dumps(data)
    assert substr in data.get("message", data.get("Message", "")), data


def umsg(*blocks):
    return {"role": "user", "content": list(blocks)}


def amsg(*blocks):
    return {"role": "assistant", "content": list(blocks)}


def cfg_body(**cfg):
    return {"messages": [U], "inferenceConfig": cfg}


def tools_body(*tools):
    return {"messages": [U], "toolConfig": {"tools": list(tools)}}


def spec(name, **kw):
    return {"toolSpec": {"name": name, **kw}}


SCHEMA = {"inputSchema": {"json": {}}}
TOOL_RESULT_CONV = [U, amsg(USE)]

CONVERSE_CASES = [
    ({"messages": []}, START),
    ({"messages": "x"}, START),
    ({}, START),
    ({"messages": [A, U]}, START),
    ({"messages": ["x"]}, "messages.0 must be an object"),
    ({"messages": [U, {"role": "system", "content": [{"text": "x"}]}]}, "messages.1.role must be one of [user, assistant]"),
    ({"messages": [U, U]}, ALT),
    ({"messages": [U, A, A]}, ALT),
    ({"messages": [{"role": "user"}]}, EMPTY),
    ({"messages": [{"role": "user", "content": []}]}, EMPTY),
    ({"messages": [{"role": "user", "content": "hi"}]}, EMPTY),
    ({"messages": [umsg({"text": "ok"}, {"text": "  "})]}, "ContentBlock object at messages.0.content.1 is blank"),
    ({"messages": TOOL_RESULT_CONV + [umsg(RES, RES)], "toolConfig": TC},
     "The number of toolResult blocks at messages.2.content exceeds"),
    ({"messages": TOOL_RESULT_CONV + [umsg({"toolResult": {"toolUseId": "zz", "content": [{"text": "r"}]}})],
      "toolConfig": TC}, "The toolUseId zz in messages.2.content.0 doesn't match any toolUse block"),
    ({"messages": TOOL_RESULT_CONV + [umsg({"text": "no result"})], "toolConfig": TC},
     "Expected toolResult blocks at messages.2.content for the following Ids: t1"),
    ({"messages": [U], "inferenceConfig": "x"}, "inferenceConfig must be an object"),
    (cfg_body(maxTokens=0), "inferenceConfig.maxTokens must be an integer greater than or equal to 1."),
    (cfg_body(maxTokens="5"), "inferenceConfig.maxTokens must be an integer"),
    (cfg_body(maxTokens=True), "inferenceConfig.maxTokens must be an integer"),
    (cfg_body(temperature=1.5), "inferenceConfig.temperature must be between 0 and 1."),
    (cfg_body(temperature=-0.1), "inferenceConfig.temperature must be between 0 and 1."),
    (cfg_body(temperature=True), "inferenceConfig.temperature must be between 0 and 1."),
    (cfg_body(topP="a"), "inferenceConfig.topP must be between 0 and 1."),
    (cfg_body(topP=2), "inferenceConfig.topP must be between 0 and 1."),
    (cfg_body(stopSequences="x"), "inferenceConfig.stopSequences must be a list of strings."),
    (cfg_body(stopSequences=["a", 1]), "inferenceConfig.stopSequences must be a list of strings."),
    ({"messages": TOOL_RESULT_CONV + [umsg(RES)]}, "The toolConfig field must be defined when using toolUse and toolResult"),
    ({"messages": [U], "toolConfig": {}}, "toolConfig.tools must contain at least one tool"),
    (tools_body(), "toolConfig.tools must contain at least one tool"),
    ({"messages": [U], "toolConfig": {"tools": "x"}}, "toolConfig.tools must contain at least one tool"),
    (tools_body(spec("ok", **SCHEMA), spec("bad name!", **SCHEMA)),
     "The tool name 'bad name!' at toolConfig.tools.1 is invalid. Tool names must match [a-zA-Z0-9_-]{1,64}."),
    (tools_body(spec(None, **SCHEMA)), "The tool name None at toolConfig.tools.0 is invalid"),
    (tools_body(spec("a" * 65, **SCHEMA)), "is invalid"),
    (tools_body(TOOL, TOOL), "Tool names must be unique: get_x"),
    (tools_body(spec("x")), "toolConfig.tools.0.toolSpec.inputSchema.json must be an object"),
    (tools_body(spec("x", inputSchema={"json": []})), "toolConfig.tools.0.toolSpec.inputSchema.json must be an object"),
]


@pytest.mark.parametrize("body,substr", CONVERSE_CASES)
def test_converse_rejected(mock, body, substr):
    expect_400(mock, "converse", body, substr)


@pytest.mark.parametrize("body,substr", [CONVERSE_CASES[3], CONVERSE_CASES[16], CONVERSE_CASES[33]])
def test_converse_stream_rejected_before_stream(mock, body, substr):
    expect_400(mock, "converse-stream", body, substr)


def test_converse_rejected_via_boto(br):
    with pytest.raises(ClientError) as e:
        br.converse(modelId=CLAUDE, messages=[A, U])
    assert e.value.response["ResponseMetadata"]["HTTPStatusCode"] == 400
    assert e.value.response["Error"]["Code"] == "ValidationException"
    assert START in e.value.response["Error"]["Message"]


def bad(x):
    return f"Malformed input request: {x}, please reformat your input and try again."


IU = {"role": "user", "content": "hi"}
IA = {"role": "assistant", "content": "yo"}
IUSE = {"type": "tool_use", "id": "t1", "name": "get_x", "input": {}}
IRES = {"type": "tool_result", "tool_use_id": "t1", "content": "r"}
NONEMPTY = "messages: text content blocks must be non-empty"
BADCONTENT = "messages.0: content must be a non-empty string or list of blocks"


def without(k):
    b = invoke_body()
    del b[k]
    return b


def msgs(*ms):
    return invoke_body(messages=list(ms))


INVOKE_CASES = [
    (without("anthropic_version"), bad("#: required key [anthropic_version] not found")),
    (without("max_tokens"), bad("#: required key [max_tokens] not found")),
    (without("messages"), bad("#: required key [messages] not found")),
    ({"messages": [IU]}, bad("#: required key [anthropic_version] not found")),
    (invoke_body(foo=1), bad("#: extraneous key [foo] is not permitted")),
    (invoke_body(anthropic_version="v1"), bad("#/anthropic_version: 'v1' is not a valid enum value")),
    (invoke_body(max_tokens=0), bad("#/max_tokens: 0 is not greater or equal to 1")),
    (invoke_body(max_tokens="5"), bad("#/max_tokens: 5 is not greater or equal to 1")),
    (invoke_body(temperature="hot"), bad("#/temperature: expected type: Number, found: str")),
    (invoke_body(top_p=None), bad("#/top_p: expected type: Number, found: NoneType")),
    (invoke_body(temperature=-1), bad("#/temperature: -1 is not greater or equal to 0")),
    (invoke_body(top_p=1.5), bad("#/top_p: 1.5 is not less or equal to 1.0")),
    (invoke_body(top_k=-1), bad("#/top_k: -1 is not greater or equal to 0")),
    (invoke_body(top_k=1.5), bad("#/top_k: 1.5 is not greater or equal to 0")),
    (msgs(), "messages: at least one message is required"),
    (invoke_body(messages="x"), "messages: at least one message is required"),
    (msgs(IA, IU), 'messages: first message must use the "user" role'),
    (msgs("x"), "messages.0 must be an object"),
    (msgs(IU, {"role": "tool", "content": "x"}), "messages.1.role: Input should be 'user' or 'assistant'"),
    (msgs(IU, IU), 'messages: roles must alternate between "user" and "assistant", but found multiple "user" roles in a row'),
    (msgs(IU, IA, IA), 'found multiple "assistant" roles in a row'),
    (msgs({"role": "user", "content": "  "}), NONEMPTY),
    (msgs({"role": "user", "content": [{"type": "text", "text": ""}]}), NONEMPTY),
    (msgs({"role": "user", "content": []}), BADCONTENT),
    (msgs({"role": "user", "content": 5}), BADCONTENT),
    (msgs(IU, {"role": "assistant", "content": [IUSE]}, {"role": "user", "content": [{**IRES, "tool_use_id": "zz"}]}),
     "messages.2.content.0: unexpected `tool_use_id` found in `tool_result` blocks: zz. Each `tool_result` block must "
     "have a corresponding `tool_use` block in the previous message."),
    (msgs(IU, {"role": "assistant", "content": [IUSE]}, {"role": "user", "content": "nope"}),
     "messages.2: `tool_use` ids were found without `tool_result` blocks immediately after: t1. Each `tool_use` block "
     "must have a corresponding `tool_result` block in the next message."),
]


@pytest.mark.parametrize("body,substr", INVOKE_CASES)
def test_invoke_rejected(mock, body, substr):
    expect_400(mock, "invoke", body, substr)


def test_invoke_stream_rejected_before_stream(mock):
    expect_400(mock, "invoke-with-response-stream", invoke_body(max_tokens=0), "#/max_tokens: 0")


def test_invoke_rejected_via_boto(br):
    with pytest.raises(ClientError) as e:
        br.invoke_model(modelId=CLAUDE, body=json.dumps(invoke_body(foo=1)))
    assert e.value.response["Error"]["Code"] == "ValidationException"
    assert "extraneous key [foo]" in e.value.response["Error"]["Message"]
    with pytest.raises(ClientError) as e:
        br.invoke_model_with_response_stream(modelId=CLAUDE, body=json.dumps(invoke_body(top_k=-1)))
    assert e.value.response["ResponseMetadata"]["HTTPStatusCode"] == 400


def test_valid_converse_multi_turn_with_tools(br):
    m = [U, amsg({"text": "calling"}, USE), umsg(RES), amsg({"text": "prefill"})]
    r = br.converse(modelId=CLAUDE, messages=m, toolConfig=TC)
    assert r["ResponseMetadata"]["HTTPStatusCode"] == 200
    for cfg in ({"temperature": 0, "maxTokens": 1},
                {"temperature": 1, "topP": 1, "maxTokens": 1, "stopSequences": ["x"]}):
        assert br.converse(modelId=CLAUDE, messages=[U], inferenceConfig=cfg)["output"]


def test_valid_converse_stream(br):
    assert list(br.converse_stream(modelId=CLAUDE, messages=[U, A])["stream"])


def test_valid_invoke_anthropic(br):
    body = invoke_body(system="be brief", temperature=1, top_p=0, top_k=0, stop_sequences=["x"],
                       tools=[{"name": "get_x", "description": "d", "input_schema": {"type": "object"}}],
                       messages=[IU, {"role": "assistant", "content": [{"type": "text", "text": "ok"}, IUSE]},
                                 {"role": "user", "content": [IRES, {"type": "text", "text": "thanks"}]},
                                 {"role": "assistant", "content": "prefill"}])
    r = br.invoke_model(modelId=CLAUDE, body=json.dumps(body))
    assert r["ResponseMetadata"]["HTTPStatusCode"] == 200
    assert json.loads(r["body"].read())


def test_converse_cache_point_tool(br):
    tc = {"tools": [TOOL, {"cachePoint": {"type": "default"}}]}
    assert br.converse(modelId=CLAUDE, messages=[U], toolConfig=tc)["output"]


NOVA = "amazon.nova-lite-v1:0"
NOVA_BODY = {"schemaVersion": "messages-v1", "messages": [{"role": "user", "content": [{"text": "hi"}]}],
             "inferenceConfig": {"maxTokens": 10}}


@pytest.mark.parametrize("api", ["invoke", "invoke-with-response-stream"])
def test_non_anthropic_invoke_not_strict_checked(mock, api):
    with MockBedrock() as plain:
        want = raw_post(plain, api, NOVA_BODY, NOVA)[0]
    status, text = raw_post(mock, api, NOVA_BODY, NOVA)
    assert status == want and "Malformed input request" not in text


@pytest.mark.parametrize("api", ["invoke", "invoke-with-response-stream"])
def test_anthropic_model_id_without_version_rejected(mock, api):
    status, text = raw_post(mock, api, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
                           "us.anthropic.claude-test")
    assert status == 400 and "anthropic_version" in text


def test_embedding_not_strict_checked(br):
    r = br.invoke_model(modelId=TITAN, body=json.dumps({"inputText": "hello"}))
    assert json.loads(r["body"].read())["embedding"]


def test_rejected_request_logged_400(mock):
    before = len(mock.requests())
    assert post(mock, "converse", {"messages": [A]})[0] == 400
    rows = mock.requests()
    assert len(rows) == before + 1 and rows[-1]["status"] == 400 and rows[-1]["api"] == "converse"
    post(mock, "invoke", invoke_body(max_tokens=0))
    assert mock.requests()[-1]["status"] == 400


def test_strict_off_accepts_invalid_bodies():
    with MockBedrock() as m:
        assert post(m, "converse", {"messages": [A]})[0] == 200
        assert post(m, "invoke", invoke_body(messages=[IA]))[0] == 200


def _cmsg(*blocks_by_role):
    return {"messages": [{"role": r, "content": b} for r, b in blocks_by_role]}


MALFORMED_CONVERSE = {
    "text_int": _cmsg(("user", [{"text": 123}])),
    "use_id_list": _cmsg(("user", [{"text": "hi"}]),
                         ("assistant", [{"toolUse": {"toolUseId": ["a"], "name": "get_x", "input": {}}}]),
                         ("user", [{"toolResult": {"toolUseId": "a", "content": [{"text": "r"}]}}])),
    "result_id_list": _cmsg(("user", [{"text": "hi"}]),
                            ("assistant", [{"toolUse": {"toolUseId": "a", "name": "get_x", "input": {}}}]),
                            ("user", [{"toolResult": {"toolUseId": ["a"], "content": [{"text": "r"}]}}])),
    "tools_int": {"messages": [{"role": "user", "content": [{"text": "hi"}]}], "toolConfig": {"tools": 5}},
    "toolconfig_not_object": {"messages": [{"role": "user", "content": [{"text": "hi"}]}], "toolConfig": 5},
    "result_text_int": _cmsg(("user", [{"text": "hi"}]),
                             ("assistant", [{"toolUse": {"toolUseId": "a", "name": "get_x", "input": {}}}]),
                             ("user", [{"toolResult": {"toolUseId": "a", "content": [{"text": 123}]}}])),
}
MALFORMED_INVOKE = {
    "text_int": invoke_body(messages=[{"role": "user", "content": [{"type": "text", "text": 123}]}]),
    "use_id_list": invoke_body(messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": ["a"], "name": "get_x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "r"}]}]),
    "result_id_list": invoke_body(messages=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "get_x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": ["a"], "content": "r"}]}]),
    "content_int": invoke_body(messages=[{"role": "user", "content": 123}]),
    "tools_list_of_int": invoke_body(tools=[1]),
    "tools_int": invoke_body(tools=5),
}
MALFORMED_CASES = (
    [(p, k, b) for p in ("converse", "converse-stream") for k, b in MALFORMED_CONVERSE.items()]
    + [(p, k, b) for p in ("invoke", "invoke-with-response-stream") for k, b in MALFORMED_INVOKE.items()])


def _check_malformed_400(m, path, body):
    before = len(m.requests())
    status, etype, data = post(m, path, body)
    assert status == 400, data
    assert etype == "ValidationException"
    rows = m.requests()
    assert len(rows) == before + 1 and rows[-1]["status"] == 400


@pytest.mark.parametrize("path,kind,body", MALFORMED_CASES)
def test_malformed_nested_values_400_strict(mock, path, kind, body):
    _check_malformed_400(mock, path, body)


@pytest.mark.parametrize("path,kind,body", MALFORMED_CASES)
def test_malformed_nested_values_400_non_strict(path, kind, body):
    with MockBedrock() as m:
        _check_malformed_400(m, path, body)
