import json
import unittest
import urllib.error
import urllib.request

from mock_bedrock.testing import MockBedrock

try:
    import boto3  # noqa: F401
    from botocore.config import Config
except ImportError:
    boto3 = None

CHAT = "anthropic.claude-test"
CONVERSE = {
    "system": [{"text": "be brief"}],
    "messages": [{"role": "user", "content": [{"text": "weather?"}]}],
    "inferenceConfig": {"maxTokens": 50, "temperature": 0.2},
    "toolConfig": {"tools": [{"toolSpec": {"name": "get_weather", "description": "w",
                                           "inputSchema": {"json": {"type": "object"}}}}]},
}


def post(mock, path, data):
    req = urllib.request.Request(mock.endpoint_url + path, method="POST", data=data,
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
            return r.status, r.headers
    except urllib.error.HTTPError as e:
        e.read()
        return e.code, e.headers


class CaptureTests(unittest.TestCase):
    def test_converse_body_captured(self):
        with MockBedrock() as mock:
            status, _ = post(mock, f"/model/{CHAT}/converse", json.dumps(CONVERSE).encode())
            self.assertEqual(status, 200)
            row = mock.last_request()
            self.assertEqual(row["body"], CONVERSE)
            self.assertFalse(row["stream"])
            self.assertIsNone(row["scenario"])
            self.assertNotIn("raw", row)

    @unittest.skipIf(boto3 is None, "boto3 missing")
    def test_boto3_request_id(self):
        with MockBedrock() as mock:
            br = mock.client(config=Config(retries={"max_attempts": 1}))
            r = br.converse(modelId=CHAT, **CONVERSE)
            self.assertEqual(r["ResponseMetadata"]["RequestId"], mock.last_request()["request_id"])

    def test_stream_row(self):
        with MockBedrock() as mock:
            status, headers = post(mock, f"/model/{CHAT}/converse-stream", json.dumps(CONVERSE).encode())
            self.assertEqual(status, 200)
            row = mock.last_request()
            self.assertTrue(row["stream"])
            self.assertEqual(row["request_id"], headers["x-amzn-RequestId"])

    def test_error_response_header_matches(self):
        with MockBedrock() as mock:
            status, headers = post(mock, f"/model/{CHAT}/converse", b"nope")
            self.assertEqual(status, 400)
            self.assertEqual(mock.last_request()["request_id"], headers["x-amzn-RequestId"])

    def test_seq_and_reset(self):
        with MockBedrock() as mock:
            for _ in range(3):
                post(mock, f"/model/{CHAT}/converse", json.dumps(CONVERSE).encode())
            self.assertEqual([r["seq"] for r in mock.requests()], [1, 2, 3])
            mock.reset()
            post(mock, f"/model/{CHAT}/converse", json.dumps(CONVERSE).encode())
            self.assertEqual([r["seq"] for r in mock.requests()], [1])

    def test_malformed_body(self):
        with MockBedrock() as mock:
            status, _ = post(mock, f"/model/{CHAT}/converse", b"{not json")
            self.assertEqual(status, 400)
            row = mock.last_request()
            self.assertIsNone(row["body"])
            self.assertEqual(row["raw"], "{not json")
            self.assertEqual(row["status"], 400)
            self.assertIsNone(row["scenario"])
            post(mock, f"/model/{CHAT}/converse", b"x" * 5000)
            self.assertEqual(len(mock.last_request()["raw"]), 2000)

    def test_embedding_row(self):
        with MockBedrock() as mock:
            body = {"inputText": "hello"}
            status, _ = post(mock, "/model/amazon.titan-embed-text-v2:0/invoke", json.dumps(body).encode())
            self.assertEqual(status, 200)
            row = mock.last_request()
            self.assertEqual(row["body"], body)
            self.assertEqual(row["texts"], 1)
            self.assertEqual(row["family"], "embedding")
            self.assertTrue(row["request_id"])

    def test_jsonl_matches_memory(self):
        with MockBedrock() as mock:
            post(mock, f"/model/{CHAT}/converse", json.dumps(CONVERSE).encode())
            post(mock, f"/model/{CHAT}/converse", b"bad")
            lines = (mock.log_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(x) for x in lines], mock.requests())

    def test_filter_by_scenario(self):
        sc = [{"id": "boom", "match": {"prompt": "weather"}, "reply": {"text": "x"}}]
        with MockBedrock(scenarios=sc) as mock:
            post(mock, f"/model/{CHAT}/converse", json.dumps(CONVERSE).encode())
            post(mock, f"/model/{CHAT}/converse",
                 json.dumps({"messages": [{"role": "user", "content": [{"text": "other"}]}]}).encode())
            self.assertEqual(len(mock.requests()), 2)
            rows = mock.requests(scenario="boom")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["scenario"], "boom")


if __name__ == "__main__":
    unittest.main()
