import unittest

from mock_bedrock import __version__
from mock_bedrock.server import main
from tests.helpers import Harness

INV = {"max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}
CONV = {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}
EMB = {"texts": ["a", "b"]}

SEQ = [("/model/anthropic.x/invoke", INV), ("/model/anthropic.x/invoke-with-response-stream", INV),
       ("/model/anthropic.x/converse", CONV), ("/model/cohere.embed-english-v3/invoke", EMB),
       ("/model/anthropic.x/invoke", INV)]


def run(h):
    out = []
    for path, body in SEQ:
        status, headers, data = h.request(path, body)
        out.append((status, headers["x-amzn-RequestId"], data))
    return out


class TestDeterministic(unittest.TestCase):
    def test_same_sequence_same_output(self):
        a, b = Harness(deterministic=True), Harness(deterministic=True)
        try:
            ra, rb = run(a), run(b)
        finally:
            a.stop()
            b.stop()
        self.assertEqual(ra, rb)
        self.assertEqual([r[1] for r in ra], [f"00000000-0000-4000-8000-{i:012d}" for i in range(1, 6)])
        self.assertEqual(ra[0][2]["id"], "msg_mock_0000000000000001")
        self.assertEqual(ra[4][2]["id"], "msg_mock_0000000000000003")  # msg ids: invoke, stream, invoke
        self.assertEqual(ra[3][2]["id"], "00000000-0000-4000-8000-000000000001")
        self.assertIsInstance(ra[1][2], bytes)  # event stream

    def test_log_matches_header(self):
        h = Harness(deterministic=True)
        try:
            _, headers, _ = h.request("/model/anthropic.x/invoke", INV)
            rid = headers["x-amzn-RequestId"]
            self.assertEqual(h.requests_log()[-1]["request_id"], rid)
        finally:
            h.stop()

    def test_reset_restarts_counters(self):
        h = Harness(deterministic=True)
        try:
            first = h.request("/model/anthropic.x/invoke", INV)
            h.request("/model/anthropic.x/invoke", INV)
            self.assertEqual(h.request("/_fakerock/reset", {})[0], 200)
            again = h.request("/model/anthropic.x/invoke", INV)
            self.assertEqual(first[1]["x-amzn-RequestId"], again[1]["x-amzn-RequestId"])
            self.assertEqual(first[2]["id"], again[2]["id"])
        finally:
            h.stop()

    def test_reset_restarts_tool_use_ids(self):
        conv = {"messages": [{"role": "user", "content": [{"text": "hi"}]}],
                "toolConfig": {"tools": [{"toolSpec": {"name": "t", "inputSchema": {"json": {"type": "object"}}}}]}}
        inv = dict(INV, tools=[{"name": "t", "input_schema": {"type": "object"}}])
        h = Harness(deterministic=True)
        try:
            def conv_id():
                return h.request("/model/anthropic.x/converse", conv)[2]["output"]["message"]["content"][0]["toolUse"]["toolUseId"]

            def inv_id():
                return [b for b in h.request("/model/anthropic.x/invoke", inv)[2]["content"] if b["type"] == "tool_use"][0]["id"]
            self.assertEqual(conv_id(), "tooluse_mock_1")
            self.assertEqual(conv_id(), "tooluse_mock_2")
            self.assertEqual(h.request("/_fakerock/reset", {})[0], 200)
            self.assertEqual(conv_id(), "tooluse_mock_1")
            self.assertEqual(inv_id(), "tooluse_mock_2")
            h.request("/_fakerock/reset", {})
            self.assertEqual(inv_id(), "tooluse_mock_1")
        finally:
            h.stop()

    def test_default_is_random(self):
        h = Harness()
        try:
            a = h.request("/model/anthropic.x/invoke", INV)
            b = h.request("/model/anthropic.x/invoke", INV)
        finally:
            h.stop()
        self.assertNotEqual(a[1]["x-amzn-RequestId"], b[1]["x-amzn-RequestId"])
        self.assertNotEqual(a[2]["id"], b[2]["id"])
        self.assertNotEqual(a[1]["x-amzn-RequestId"], "00000000-0000-4000-8000-000000000001")

    def test_version(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            main(["--version"])
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(buf.getvalue().strip(), f"mock_bedrock {__version__}")
        self.assertEqual(__version__, "0.4.0")


if __name__ == "__main__":
    unittest.main()
