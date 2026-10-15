"""New fault modes: extra error types, drop, _latency_seconds, _chunk_delay_seconds."""
import json
import time

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from urllib3.exceptions import HTTPError as Urllib3Error

from mock_bedrock.control import ERROR_MODES
from mock_bedrock.testing import MockBedrock

CLAUDE = "anthropic.claude-3-haiku-20240307-v1:0"
MSGS = [{"role": "user", "content": [{"text": "please summarise this rather long sentence for me"}]}]
INVOKE_BODY = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 50,
                          "messages": [{"role": "user", "content": "hello there, how are you"}]})
NEW_MODES = ["model_timeout", "model_not_ready", "access_denied", "not_found", "quota_exceeded", "model_error"]
CFG = Config(retries={"total_max_attempts": 1, "mode": "standard"}, read_timeout=5, connect_timeout=5)


@pytest.fixture
def mock():
    with MockBedrock() as m:
        yield m


@pytest.fixture
def br(mock):
    return mock.client(config=CFG)


def converse(br):
    return br.converse(modelId=CLAUDE, messages=MSGS)


def invoke(br):
    return br.invoke_model(modelId=CLAUDE, body=INVOKE_BODY)


def test_expected_status_codes():
    assert ERROR_MODES["model_timeout"][0] == 408
    assert ERROR_MODES["model_not_ready"][0] == 429
    assert ERROR_MODES["access_denied"][0] == 403
    assert ERROR_MODES["not_found"][0] == 404
    assert ERROR_MODES["quota_exceeded"][0] == 400
    assert ERROR_MODES["model_error"][0] == 424


@pytest.mark.parametrize("mode", NEW_MODES)
@pytest.mark.parametrize("call", [converse, invoke])
def test_error_modes(mock, br, mode, call):
    status, etype, _ = ERROR_MODES[mode]
    mock.set_modes({"chat": {"_default": mode}})
    with pytest.raises(ClientError) as ei:
        call(br)
    assert ei.value.response["Error"]["Code"] == etype
    assert ei.value.response["ResponseMetadata"]["HTTPStatusCode"] == status


@pytest.mark.parametrize("model", ["cohere.embed-english-v3", "amazon.titan-embed-text-v2:0"])
def test_error_mode_embeddings(mock, br, model):
    mock.set_modes({"embedding": {"_default": "access_denied"}})
    body = json.dumps({"texts": ["hi"], "input_type": "search_document"} if model.startswith("cohere")
                      else {"inputText": "hi"})
    with pytest.raises(ClientError) as ei:
        br.invoke_model(modelId=model, body=body)
    assert ei.value.response["Error"]["Code"] == "AccessDeniedException"
    assert ei.value.response["ResponseMetadata"]["HTTPStatusCode"] == 403


def test_converse_stream_error_mode_before_stream(mock, br):
    mock.set_modes({"chat": {"_default": "model_timeout"}})
    with pytest.raises(ClientError) as ei:
        br.converse_stream(modelId=CLAUDE, messages=MSGS)
    assert ei.value.response["Error"]["Code"] == "ModelTimeoutException"
    assert ei.value.response["ResponseMetadata"]["HTTPStatusCode"] == 408


def test_scenario_access_denied():
    with MockBedrock(scenarios=[{"id": "deny", "reply": {"mode": "access_denied"}, "times": 1}]) as mock:
        br = mock.client(config=CFG)
        with pytest.raises(ClientError) as ei:
            converse(br)
        assert ei.value.response["Error"]["Code"] == "AccessDeniedException"
        assert converse(br)["output"]["message"]  # times exhausted -> normal answer
        assert mock.requests(scenario="deny")


def test_latency_seconds(mock, br):
    mock.set_modes({"_latency_seconds": 0.2})
    t = time.monotonic()
    converse(br)
    assert time.monotonic() - t >= 0.2
    mock.set_modes({"_latency_seconds": 0.2, "chat": {"_default": "throttle"}})
    t = time.monotonic()
    with pytest.raises(ClientError):
        converse(br)
    assert time.monotonic() - t >= 0.2


def test_latency_seconds_embeddings(mock, br):
    mock.set_modes({"_latency_seconds": 0.2})
    t = time.monotonic()
    br.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=json.dumps({"inputText": "hi"}))
    assert time.monotonic() - t >= 0.2


def test_chunk_delay_seconds(mock, br):
    mock.set_modes({"_chunk_delay_seconds": 0.1})
    t = time.monotonic()
    events = list(br.converse_stream(modelId=CLAUDE, messages=MSGS)["stream"])
    assert len(events) >= 4
    assert time.monotonic() - t >= 0.2


@pytest.mark.parametrize("key", ["_latency_seconds", "_chunk_delay_seconds", "_stall_seconds"])
@pytest.mark.parametrize("bad", [-1, "1", True])
def test_invalid_numeric_keys(mock, key, bad):
    with pytest.raises(RuntimeError):  # control client turns the 400 into RuntimeError
        mock.set_modes({key: bad})


def test_invalid_numeric_keys_at_startup():
    with pytest.raises(ValueError):
        MockBedrock(modes={"_latency_seconds": -1}).start()


CONN_ERRORS = BotoCoreError  # non-streaming: botocore wraps it (ConnectionClosedError / HTTPClientError)
# Mid-stream, botocore does not wrap read errors: urllib3's ProtocolError (IncompleteRead) escapes.
STREAM_ERRORS = (BotoCoreError, Urllib3Error)


@pytest.mark.parametrize("call", [converse, invoke])
def test_drop_non_streaming(mock, br, call):
    mock.set_modes({"chat": {"_default": "drop"}})
    with pytest.raises(CONN_ERRORS):
        call(br)
    assert mock.requests()[-1]["status"] == 0
    mock.set_modes({})
    assert call(br)  # still serving


def test_drop_embedding(mock, br):
    mock.set_modes({"embedding": {"_default": "drop"}})
    with pytest.raises(CONN_ERRORS):
        br.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=json.dumps({"inputText": "hi"}))
    assert mock.requests()[-1]["status"] == 0


def test_drop_converse_stream(mock, br):
    mock.set_modes({"chat": {"_default": "drop"}})
    with pytest.raises(STREAM_ERRORS):
        resp = br.converse_stream(modelId=CLAUDE, messages=MSGS)
        list(resp["stream"])
    assert mock.requests(api="converse-stream")[-1]["status"] == 0
    mock.set_modes({})
    assert list(br.converse_stream(modelId=CLAUDE, messages=MSGS)["stream"])


def test_drop_invoke_stream(mock, br):
    mock.set_modes({"chat": {"_default": "drop"}})
    with pytest.raises(STREAM_ERRORS):
        resp = br.invoke_model_with_response_stream(modelId=CLAUDE, body=INVOKE_BODY)
        list(resp["body"])
    assert mock.requests(api="invoke-stream")[-1]["status"] == 0


def test_drop_via_scenario_and_stream_error_still_works():
    with MockBedrock(scenarios=[{"id": "d", "reply": {"mode": "drop"}}]) as mock:
        br = mock.client(config=CFG)
        with pytest.raises(CONN_ERRORS):
            converse(br)
        assert mock.requests(scenario="d")[-1]["status"] == 0
        mock.set_scenarios([{"reply": {"mode": "stream_error"}}])
        with pytest.raises(Exception) as ei:
            list(br.converse_stream(modelId=CLAUDE, messages=MSGS)["stream"])
        assert "ModelStreamError" in type(ei.value).__name__ or "modelStreamError" in str(ei.value)
