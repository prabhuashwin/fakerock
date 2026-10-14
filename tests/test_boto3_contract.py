import json
import unittest

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover
    boto3 = None

from tests.helpers import Harness


@unittest.skipIf(boto3 is None, "boto3 not installed")
class Boto3Contract(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.addCleanup(self.h.stop)
        self.client = boto3.client(
            "bedrock-runtime", region_name="us-east-1", endpoint_url=f"http://127.0.0.1:{self.h.port}",
            aws_access_key_id="dummy", aws_secret_access_key="dummy",
            config=Config(retries={"max_attempts": 1}))

    def test_claude(self):
        r = self.client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 50,
                             "messages": [{"role": "user", "content": "hi"}]}))
        out = json.loads(r["body"].read())
        self.assertEqual(out["content"][0]["text"], "Mock reply to: hi")

    def test_cohere(self):
        r = self.client.invoke_model(modelId="cohere.embed-english-v3",
                                     body=json.dumps({"texts": ["a cat"], "input_type": "search_query"}))
        out = json.loads(r["body"].read())
        self.assertEqual(len(out["embeddings"][0]), 1024)

    def test_converse_and_streams_smoke(self):
        m = "anthropic.claude-3-haiku-20240307-v1:0"
        msgs = [{"role": "user", "content": [{"text": "hi"}]}]
        r = self.client.converse(modelId=m, messages=msgs)
        self.assertEqual(r["output"]["message"]["content"][0]["text"], "Mock reply to: hi")
        r = self.client.converse_stream(modelId=m, messages=msgs)
        self.assertEqual("".join(e["contentBlockDelta"]["delta"]["text"]
                                 for e in r["stream"] if "contentBlockDelta" in e), "Mock reply to: hi")
        r = self.client.invoke_model_with_response_stream(
            modelId=m, body=json.dumps({"max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}))
        evs = [json.loads(e["chunk"]["bytes"]) for e in r["body"]]
        self.assertEqual(evs[-1]["type"], "message_stop")

    def test_throttle(self):
        self.h.set_modes({"chat": {"_default": "throttle"}})
        with self.assertRaises(ClientError) as cm:
            self.client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({"max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}))
        self.assertEqual(cm.exception.response["Error"]["Code"], "ThrottlingException")


if __name__ == "__main__":
    unittest.main()
