import json
import time
import unittest
from unittest import mock

try:
    import boto3  # noqa: F401
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError:  # pragma: no cover
    boto3 = None

from mock_bedrock.testing import MockBedrock

TITAN = "amazon.titan-embed-text-v2:0"
COHERE = "cohere.embed-english-v3"


def body_for(model, text):
    if model == TITAN:
        return json.dumps({"inputText": text})
    return json.dumps({"texts": [text], "input_type": "search_document"})


def is_ok(model, resp):
    data = json.loads(resp["body"].read())
    return "embedding" in data if model == TITAN else "embeddings" in data


@unittest.skipIf(boto3 is None, "boto3 not installed")
class EmbeddingControl(unittest.TestCase):
    def make(self, attempts=1, **kw):
        m = MockBedrock(**kw)
        m.start()
        self.addCleanup(m.stop)
        cfg = Config(retries={"total_max_attempts": attempts})
        return m, m.client(config=cfg)

    def invoke(self, br, model, text="hello"):
        return br.invoke_model(modelId=model, body=body_for(model, text))

    def code(self, br, model, text="hello"):
        with self.assertRaises(ClientError) as cm:
            self.invoke(br, model, text)
        return cm.exception.response["Error"]["Code"], cm.exception.response["ResponseMetadata"]["HTTPStatusCode"]

    def test_default_mode_throttle(self):
        for model in (TITAN, COHERE):
            with self.subTest(model=model):
                m, br = self.make(modes={"embedding": {"_default": "throttle"}})
                self.assertEqual(self.code(br, model), ("ThrottlingException", 429))
                row = m.requests()[-1]
                self.assertEqual((row["family"], row["mode"], row["status"]), ("embedding", "throttle", 429))

    def test_stall_answers_after_sleep(self):
        for model in (TITAN, COHERE):
            with self.subTest(model=model):
                m, br = self.make(modes={"_stall_seconds": 0.2, "embedding": {"_default": "stall"}})
                t0 = time.monotonic()
                self.assertTrue(is_ok(model, self.invoke(br, model)))
                self.assertGreaterEqual(time.monotonic() - t0, 0.19)

    def test_stream_error_mode(self):
        for model in (TITAN, COHERE):
            with self.subTest(model=model):
                m, br = self.make(modes={"embedding": {"_default": "stream_error"}})
                self.assertEqual(self.code(br, model), ("InternalServerException", 500))
                self.assertEqual(len(m.requests()), 1)

    def test_mode_needle(self):
        for model in (TITAN, COHERE):
            with self.subTest(model=model):
                m, br = self.make(modes={"embedding": {"secret": "validation"}})
                self.assertTrue(is_ok(model, self.invoke(br, model, "plain")))
                self.assertEqual(self.code(br, model, "a secret text"), ("ValidationException", 400))
                rows = m.requests()
                self.assertEqual([r["mode"] for r in rows], ["answer", "validation"])

    def test_scenario_mode_times_once(self):
        for model in (TITAN, COHERE):
            with self.subTest(model=model):
                scen = [{"id": "u", "match": {"model": "titan-embed|cohere"}, "times": 1,
                         "reply": {"mode": "unavailable"}}]
                m, br = self.make(scenarios=scen)
                self.assertEqual(self.code(br, model), ("ServiceUnavailableException", 503))
                self.assertTrue(is_ok(model, self.invoke(br, model)))
                rows = m.requests()
                self.assertEqual([(r["mode"], r["scenario"], r["status"]) for r in rows],
                                 [("scenario", "u", 503), ("answer", None, 200)])
                self.assertTrue(all(r["family"] == "embedding" for r in rows))

    def test_replies_sequence(self):
        scen = [{"id": "seq", "replies": [
            {"status": 500, "error_type": "InternalServerException", "message": "boom"},
            {"text": "ignored"}]}]
        for model in (TITAN, COHERE):
            with self.subTest(model=model):
                m, br = self.make(scenarios=scen)
                self.assertEqual(self.code(br, model), ("InternalServerException", 500))
                self.assertTrue(is_ok(model, self.invoke(br, model)))
                rows = m.requests()
                self.assertEqual([(r["mode"], r["scenario"], r["status"]) for r in rows],
                                 [("scenario", "seq", 500), ("scenario", "seq", 200)])

    def test_scenario_prompt_regex(self):
        scen = [{"id": "p", "match": {"prompt": r"bad\d+"}, "reply": {"mode": "throttle"}}]
        for model in (TITAN, COHERE):
            with self.subTest(model=model):
                m, br = self.make(scenarios=scen)
                self.assertTrue(is_ok(model, self.invoke(br, model, "fine")))
                self.assertEqual(self.code(br, model, "this is bad42"), ("ThrottlingException", 429))
                self.assertEqual(m.requests(scenario="p")[0]["status"], 429)

    def test_malformed_body_400_and_no_scenario_consumed(self):
        scen = [{"id": "once", "times": 1, "reply": {"mode": "throttle"}}]
        for model in (TITAN, COHERE):
            with self.subTest(model=model):
                m, br = self.make(scenarios=scen)
                with self.assertRaises(ClientError) as cm:
                    br.invoke_model(modelId=model, body=json.dumps({"nonsense": 1}))
                self.assertEqual(cm.exception.response["ResponseMetadata"]["HTTPStatusCode"], 400)
                row = m.requests()[-1]
                self.assertEqual((row["family"], row["mode"], row["status"]), (None, None, 400))
                self.assertEqual(self.code(br, model), ("ThrottlingException", 429))

    def test_retries_recover(self):
        for target in ("botocore.retries.standard.ExponentialBackoff.delay_amount",
                       "botocore.retryhandler.delay_exponential"):
            p = mock.patch(target, return_value=0)
            p.start()
            self.addCleanup(p.stop)
        scen = [{"id": "t", "times": 2, "reply": {"mode": "throttle"}}]
        for model in (TITAN, COHERE):
            with self.subTest(model=model):
                m, br = self.make(attempts=3, scenarios=scen)
                r = self.invoke(br, model)
                self.assertEqual(r["ResponseMetadata"]["RetryAttempts"], 2)
                self.assertEqual(len(m.requests()), 3)


if __name__ == "__main__":
    unittest.main()
