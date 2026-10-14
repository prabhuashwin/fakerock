import json
import struct
import unittest

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, EventStreamError
except ImportError:  # pragma: no cover
    boto3 = None

try:
    from botocore.eventstream import ChecksumMismatch, EventStreamBuffer
except ImportError:  # pragma: no cover
    EventStreamBuffer = None

from mock_bedrock import eventstream
from tests.helpers import Harness

CLAUDE = "anthropic.claude-3-haiku-20240307-v1:0"
LONG = "please summarise this rather long sentence for me"
TOOL = {"tools": [{"toolSpec": {"name": "get_weather", "description": "weather",
                                "inputSchema": {"json": {"type": "object",
                                                         "properties": {"city": {"type": "string"}},
                                                         "required": ["city"]}}}}]}


@unittest.skipIf(EventStreamBuffer is None, "botocore not installed")
class EncoderTests(unittest.TestCase):
    def test_event_roundtrip(self):
        data = eventstream.event("contentBlockDelta", {"a": 1})
        buf = EventStreamBuffer()
        buf.add_data(data)
        msgs = list(buf)
        self.assertEqual(len(msgs), 1)
        m = msgs[0]
        self.assertEqual(m.headers[":event-type"], "contentBlockDelta")
        self.assertEqual(m.headers[":content-type"], "application/json")
        self.assertEqual(m.headers[":message-type"], "event")
        self.assertEqual(json.loads(m.payload), {"a": 1})
        self.assertEqual(struct.unpack(">I", data[:4])[0], len(data))

    def test_exception(self):
        buf = EventStreamBuffer()
        buf.add_data(eventstream.exception("modelStreamErrorException", "boom"))
        m = list(buf)[0]
        self.assertEqual(m.headers[":exception-type"], "modelStreamErrorException")
        self.assertEqual(m.headers[":message-type"], "exception")
        self.assertEqual(json.loads(m.payload), {"message": "boom"})

    def test_corruption_fails_crc(self):
        data = bytearray(eventstream.event("x", {"a": 1}))
        data[-6] ^= 0xFF
        buf = EventStreamBuffer()
        buf.add_data(bytes(data))
        with self.assertRaises(ChecksumMismatch):
            list(buf)


@unittest.skipIf(boto3 is None, "boto3 not installed")
class StreamingContract(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.addCleanup(self.h.stop)
        self.client = boto3.client(
            "bedrock-runtime", region_name="us-east-1", endpoint_url=f"http://127.0.0.1:{self.h.port}",
            aws_access_key_id="dummy", aws_secret_access_key="dummy",
            config=Config(retries={"max_attempts": 1}))

    def user(self, text):
        return [{"role": "user", "content": [{"text": text}]}]

    def test_converse_text(self):
        r = self.client.converse(modelId=CLAUDE, messages=self.user("hi"),
                                 system=[{"text": "be brief"}], inferenceConfig={"maxTokens": 50})
        self.assertEqual(r["output"]["message"]["content"][0]["text"], "Mock reply to: hi")
        self.assertEqual(r["stopReason"], "end_turn")
        self.assertGreater(r["usage"]["totalTokens"], 0)
        self.assertIn("be brief", self.h.requests_log()[-1]["prompt"])
        self.assertEqual(self.h.requests_log()[-1]["api"], "converse")

    def test_converse_tool_roundtrip(self):
        r = self.client.converse(modelId=CLAUDE, messages=self.user("weather?"), toolConfig=TOOL)
        self.assertEqual(r["stopReason"], "tool_use")
        block = r["output"]["message"]["content"][0]["toolUse"]
        self.assertEqual(block["name"], "get_weather")
        self.assertEqual(block["input"], {"city": "mock"})
        msgs = self.user("weather?") + [
            r["output"]["message"],
            {"role": "user", "content": [{"toolResult": {
                "toolUseId": block["toolUseId"], "content": [{"json": {"temp": 20}}], "status": "success"}}]}]
        r2 = self.client.converse(modelId=CLAUDE, messages=msgs, toolConfig=TOOL)
        self.assertEqual(r2["stopReason"], "end_turn")
        self.assertIn("text", r2["output"]["message"]["content"][0])

    def test_converse_stream_text(self):
        r = self.client.converse_stream(modelId=CLAUDE, messages=self.user(LONG))
        text, stop, deltas, usage = "", None, 0, None
        for ev in r["stream"]:
            if "contentBlockDelta" in ev:
                text += ev["contentBlockDelta"]["delta"]["text"]
                deltas += 1
            elif "messageStop" in ev:
                stop = ev["messageStop"]["stopReason"]
            elif "metadata" in ev:
                usage = ev["metadata"]["usage"]
        self.assertEqual(text, "Mock reply to: " + LONG)
        self.assertGreater(deltas, 1)
        self.assertEqual(stop, "end_turn")
        self.assertIsNotNone(usage)
        self.assertEqual(self.h.requests_log()[-1]["api"], "converse-stream")

    def test_converse_stream_tool(self):
        r = self.client.converse_stream(modelId=CLAUDE, messages=self.user("weather?"), toolConfig=TOOL)
        start, parts, stop = None, "", None
        for ev in r["stream"]:
            if "contentBlockStart" in ev:
                start = ev["contentBlockStart"]["start"]["toolUse"]
            elif "contentBlockDelta" in ev:
                parts += ev["contentBlockDelta"]["delta"]["toolUse"]["input"]
            elif "messageStop" in ev:
                stop = ev["messageStop"]["stopReason"]
        self.assertEqual(start["name"], "get_weather")
        self.assertEqual(json.loads(parts), {"city": "mock"})
        self.assertEqual(stop, "tool_use")

    def test_invoke_stream_text(self):
        body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 100,
                "messages": [{"role": "user", "content": LONG}]}
        r = self.client.invoke_model_with_response_stream(modelId=CLAUDE, body=json.dumps(body))
        text, types, metrics = "", [], None
        for ev in r["body"]:
            obj = json.loads(ev["chunk"]["bytes"])
            types.append(obj["type"])
            if obj["type"] == "content_block_delta":
                text += obj["delta"]["text"]
            if obj["type"] == "message_stop":
                metrics = obj["amazon-bedrock-invocationMetrics"]
        self.assertEqual(text, "Mock reply to: " + LONG)
        self.assertEqual(types[0], "message_start")
        self.assertEqual(types[-1], "message_stop")
        self.assertIn("outputTokenCount", metrics)
        self.assertEqual(self.h.requests_log()[-1]["api"], "invoke-stream")

    def test_stream_error_mode(self):
        self.h.set_modes({"chat": {"_default": "stream_error"}})
        r = self.client.converse_stream(modelId=CLAUDE, messages=self.user(LONG))
        seen = []
        with self.assertRaises(EventStreamError) as cm:
            for ev in r["stream"]:
                seen.append(ev)
        self.assertIn("messageStart", seen[0])
        self.assertIn("contentBlockDelta", seen[-1])
        self.assertEqual(cm.exception.response["Error"]["Code"], "modelStreamErrorException")

    def test_stream_error_on_non_stream_acts_like_error(self):
        self.h.set_modes({"chat": {"_default": "stream_error"}})
        with self.assertRaises(ClientError) as cm:
            self.client.converse(modelId=CLAUDE, messages=self.user("hi"))
        self.assertEqual(cm.exception.response["Error"]["Code"], "InternalServerException")

    def test_throttle_converse(self):
        self.h.set_modes({"chat": {"_default": "throttle"}})
        with self.assertRaises(ClientError) as cm:
            self.client.converse(modelId=CLAUDE, messages=self.user("hi"))
        self.assertEqual(cm.exception.response["Error"]["Code"], "ThrottlingException")

    def test_throttle_converse_stream_is_plain_http_error(self):
        self.h.set_modes({"chat": {"_default": "throttle"}})
        with self.assertRaises(ClientError) as cm:
            self.client.converse_stream(modelId=CLAUDE, messages=self.user("hi"))
        self.assertEqual(cm.exception.response["Error"]["Code"], "ThrottlingException")

    def test_system_and_needle_modes(self):
        self.h.set_modes({"chat": {"poison": "unavailable"}})
        ok = self.client.converse(modelId=CLAUDE, system=[{"text": "sys"}], messages=self.user("fine"))
        self.assertEqual(ok["stopReason"], "end_turn")
        with self.assertRaises(ClientError) as cm:
            self.client.converse(modelId=CLAUDE, system=[{"text": "sys"}], messages=self.user("a POISON pill"))
        self.assertEqual(cm.exception.response["Error"]["Code"], "ServiceUnavailableException")


if __name__ == "__main__":
    unittest.main()
