import json
import unittest
from unittest import mock

try:
    import boto3  # noqa: F401
    from botocore.config import Config
    from botocore.exceptions import ClientError, EventStreamError
except ImportError:  # pragma: no cover
    boto3 = None

from mock_bedrock.testing import MockBedrock

MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
MSGS = [{"role": "user", "content": [{"text": "hi"}]}]
BODY = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 50,
                   "messages": [{"role": "user", "content": "hi"}]})


def flaky(mode, times=None):
    s = {"id": "f", "reply": {"mode": mode}}
    if times:
        s["times"] = times
    return [s]


@unittest.skipIf(boto3 is None, "boto3 not installed")
class Boto3Retries(unittest.TestCase):
    retry_mode = "standard"

    def setUp(self):
        # No real backoff sleeps in these tests.
        for target in ("botocore.retries.standard.ExponentialBackoff.delay_amount",
                       "botocore.retryhandler.delay_exponential"):
            p = mock.patch(target, return_value=0)
            p.start()
            self.addCleanup(p.stop)

    def mock_client(self, scenarios):
        m = MockBedrock(scenarios=scenarios)
        m.start()
        self.addCleanup(m.stop)
        cfg = Config(retries={"mode": self.retry_mode, "total_max_attempts": 3})
        return m, m.client(config=cfg)

    def test_throttle_twice_then_answer(self):
        m, br = self.mock_client(flaky("throttle", 2))
        r = br.invoke_model(modelId=MODEL, body=BODY)
        self.assertEqual(r["ResponseMetadata"]["RetryAttempts"], 2)
        self.assertEqual(len(m.requests()), 3)

    def test_throttle_forever_gives_up(self):
        m, br = self.mock_client(flaky("throttle"))
        with self.assertRaises(ClientError) as cm:
            br.invoke_model(modelId=MODEL, body=BODY)
        self.assertEqual(cm.exception.response["Error"]["Code"], "ThrottlingException")
        self.assertEqual(len(m.requests()), 3)

    def test_unavailable_twice_then_answer(self):
        m, br = self.mock_client(flaky("unavailable", 2))
        r = br.converse(modelId=MODEL, messages=MSGS)
        self.assertEqual(r["output"]["message"]["content"][0]["text"], "Mock reply to: hi")
        self.assertEqual(len(m.requests()), 3)

    def test_internal_error_once_then_answer(self):
        m, br = self.mock_client(flaky("error", 1))
        br.converse(modelId=MODEL, messages=MSGS)
        self.assertEqual(len(m.requests()), 2)

    def test_validation_not_retried(self):
        m, br = self.mock_client(flaky("validation"))
        with self.assertRaises(ClientError) as cm:
            br.converse(modelId=MODEL, messages=MSGS)
        self.assertEqual(cm.exception.response["Error"]["Code"], "ValidationException")
        self.assertEqual(len(m.requests()), 1)

    def test_client_error_fields(self):
        scen = [{"id": "x", "reply": {"status": 429, "error_type": "ThrottlingException",
                                      "message": "slow down"}}]
        m, br = self.mock_client(scen)
        with self.assertRaises(ClientError) as cm:
            br.converse(modelId=MODEL, messages=MSGS)
        resp = cm.exception.response
        self.assertEqual(resp["Error"]["Code"], "ThrottlingException")
        self.assertEqual(resp["Error"]["Message"], "slow down")
        self.assertEqual(resp["ResponseMetadata"]["HTTPStatusCode"], 429)
        rows = m.requests()
        self.assertEqual(resp["ResponseMetadata"]["RequestId"], rows[-1]["request_id"])

    def test_stream_error(self):
        m, br = self.mock_client(flaky("stream_error"))
        r = br.converse_stream(modelId=MODEL, messages=MSGS)
        deltas = 0
        with self.assertRaises(EventStreamError) as cm:
            for ev in r["stream"]:
                if "contentBlockDelta" in ev:
                    deltas += 1
        self.assertGreaterEqual(deltas, 1)
        self.assertEqual(cm.exception.response["Error"]["Code"], "modelStreamErrorException")


class Boto3RetriesLegacy(Boto3Retries):
    retry_mode = "legacy"


if __name__ == "__main__":
    unittest.main()
