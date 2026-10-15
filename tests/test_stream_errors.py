"""Stream error modes: N deltas then an event-stream exception; fallback HTTP error when not streaming."""
import json

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.config import Config
from botocore.exceptions import ClientError, EventStreamError

from mock_bedrock.control import ERROR_MODES, STREAM_ERROR_MODES
from mock_bedrock.testing import MockBedrock

CLAUDE = "anthropic.claude-test"
TITAN = "amazon.titan-embed-text-v2:0"
COHERE = "cohere.embed-english-v3"
PROMPT = "please summarise this rather long sentence for me, thanks"  # reply > 3 chunks
MSGS = [{"role": "user", "content": [{"text": PROMPT}]}]
INVOKE_BODY = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 50,
                          "messages": [{"role": "user", "content": PROMPT}]})
CFG = Config(retries={"total_max_attempts": 1, "mode": "standard"}, read_timeout=5, connect_timeout=5)
TEXT = "Mock reply to: " + PROMPT  # default reply echoes the prompt; split into 20-char chunks
CHUNKS = [TEXT[i:i + 20] for i in range(0, len(TEXT), 20)]
MODES = list(STREAM_ERROR_MODES)


@pytest.fixture(scope="module")
def mock():
    with MockBedrock() as m:
        yield m


@pytest.fixture(autouse=True)
def _reset_modes(mock):
    mock.set_modes({})
    yield
    mock.set_modes({})


@pytest.fixture(scope="module")
def br(mock):
    return mock.client(config=CFG)


def converse_stream(br):
    """Returns (received text, EventStreamError)."""
    text, err = "", None
    try:
        for ev in br.converse_stream(modelId=CLAUDE, messages=MSGS)["stream"]:
            text += ev.get("contentBlockDelta", {}).get("delta", {}).get("text", "")
    except EventStreamError as e:
        err = e
    return text, err


def invoke_stream(br):
    text, err = "", None
    try:
        for ev in br.invoke_model_with_response_stream(modelId=CLAUDE, body=INVOKE_BODY)["body"]:
            d = json.loads(ev["chunk"]["bytes"])
            if d.get("type") == "content_block_delta":
                text += d["delta"].get("text", "")
    except EventStreamError as e:
        err = e
    return text, err


def code(err):
    return err.response["Error"]["Code"]


def expected_code(mode, call):
    exc = STREAM_ERROR_MODES[mode][0]
    return "internalServerException" if call is converse_stream and exc == "modelTimeoutException" else exc


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("call", [converse_stream, invoke_stream])
def test_stream_error_after_first_chunk(mock, br, mode, call):
    mock.set_modes({"chat": {"_default": mode}})
    text, err = call(br)
    assert err is not None and code(err) == expected_code(mode, call)
    assert text == CHUNKS[0]


@pytest.mark.parametrize("call", [converse_stream, invoke_stream])
@pytest.mark.parametrize("after,want", [(0, ""), (3, "".join(CHUNKS[:3])), (99, TEXT)])
def test_stream_error_after(mock, br, call, after, want):
    mock.set_modes({"chat": {"_default": "stream_throttle"}, "_stream_error_after": after})
    text, err = call(br)
    assert code(err) == "throttlingException"
    assert text == want


def test_model_timeout_differs_by_api(mock, br):
    mock.set_modes({"chat": {"_default": "stream_model_timeout"}})
    assert code(converse_stream(br)[1]) == "internalServerException"
    assert code(invoke_stream(br)[1]) == "modelTimeoutException"


def test_startup_modes():
    with MockBedrock(modes={"chat": {"_default": "stream_throttle"}, "_stream_error_after": 2}) as m:
        text, err = converse_stream(m.client(config=CFG))
        assert code(err) == "throttlingException" and text == CHUNKS[0] + CHUNKS[1]


@pytest.mark.parametrize("bad", [-1, 1.5, True, "2"])
def test_invalid_stream_error_after(mock, bad):
    with pytest.raises(RuntimeError):
        mock.set_modes({"_stream_error_after": bad})


@pytest.mark.parametrize("mode", MODES)
def test_non_streaming_fallback(mock, br, mode):
    mock.set_modes({"chat": {"_default": mode}})
    status, etype, _ = ERROR_MODES[STREAM_ERROR_MODES[mode][1]]
    for call in (lambda: br.converse(modelId=CLAUDE, messages=MSGS),
                 lambda: br.invoke_model(modelId=CLAUDE, body=INVOKE_BODY)):
        with pytest.raises(ClientError) as ei:
            call()
        assert ei.value.response["Error"]["Code"] == etype
        assert ei.value.response["ResponseMetadata"]["HTTPStatusCode"] == status


@pytest.mark.parametrize("model,body", [(TITAN, {"inputText": "hi"}),
                                        (COHERE, {"texts": ["hi"], "input_type": "search_document"})])
@pytest.mark.parametrize("mode,etype,status", [("stream_throttle", "ThrottlingException", 429),
                                               ("stream_unavailable", "ServiceUnavailableException", 503)])
def test_embedding_fallback(mock, br, model, body, mode, etype, status):
    mock.set_modes({"embedding": {"_default": mode}})
    with pytest.raises(ClientError) as ei:
        br.invoke_model(modelId=model, body=json.dumps(body))
    assert ei.value.response["Error"]["Code"] == etype
    assert ei.value.response["ResponseMetadata"]["HTTPStatusCode"] == status


def test_stream_throttle_as_scenario_mode(mock, br):
    mock.set_scenarios([{"reply": {"text": TEXT, "mode": "stream_throttle"}}])
    try:
        text, err = converse_stream(br)
        assert code(err) == "throttlingException" and text == CHUNKS[0]
        with pytest.raises(ClientError) as ei:
            br.converse(modelId=CLAUDE, messages=MSGS)
        assert ei.value.response["ResponseMetadata"]["HTTPStatusCode"] == 429
    finally:
        mock.set_scenarios([])
