import json
import math
import time
import unittest
import urllib.error
import urllib.request
from urllib.parse import quote

from tests.helpers import Harness

MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


def claude(text, **extra):
    return {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 100,
            "messages": [{"role": "user", "content": text}], **extra}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.addCleanup(lambda: self.h.stop())

    def invoke(self, body, model=MODEL):
        return self.h.request(f"/model/{model}/invoke", body)

    def test_health(self):
        s, h, b = self.h.request("/health", method="GET")
        self.assertEqual((s, b), (200, {"ok": True, "pack": "generic"}))
        self.assertIn("x-amzn-RequestId", h)

    def test_unknown_path(self):
        s, h, b = self.h.request("/nope", {})
        self.assertEqual(s, 404)
        self.assertEqual(h["x-amzn-ErrorType"], "UnknownOperationException")
        self.assertIn("message", b)

    def test_claude_answer(self):
        s, h, b = self.invoke(claude("hello there", system="be nice"))
        self.assertEqual(s, 200)
        self.assertEqual(b["content"][0]["text"], "Mock reply to: hello there")
        self.assertEqual(b["stop_reason"], "end_turn")
        self.assertEqual(b["role"], "assistant")
        self.assertIn("x-amzn-RequestId", h)
        self.assertGreaterEqual(b["usage"]["output_tokens"], 1)

    def test_system_blocks_and_block_content(self):
        body = claude("x", system=[{"type": "text", "text": "SYS"}])
        body["messages"] = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        self.assertEqual(self.invoke(body)[0], 200)
        self.assertTrue(self.h.requests_log()[-1]["prompt"].startswith("SYS\n"))

    def test_tools_then_tool_result(self):
        tools = [{"name": "lookup", "description": "d", "input_schema": {
            "type": "object", "required": ["q", "n", "f", "e"],
            "properties": {"q": {"type": "string"}, "n": {"type": "integer"}, "f": {"type": "boolean"},
                           "e": {"type": "string", "enum": ["a", "b"]}, "opt": {"type": "string"}}}}]
        s, _, b = self.invoke(claude("find it", tools=tools))
        self.assertEqual(s, 200)
        self.assertEqual(b["stop_reason"], "tool_use")
        tu = b["content"][0]
        self.assertEqual(tu["type"], "tool_use")
        self.assertEqual(tu["name"], "lookup")
        self.assertTrue(tu["id"].startswith("tooluse_mock_"))
        self.assertEqual(tu["input"], {"q": "mock", "n": 0, "f": False, "e": "a"})
        body = claude("find it", tools=tools)
        body["messages"] += [
            {"role": "assistant", "content": [tu]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu["id"], "content": "done"}]}]
        s, _, b = self.invoke(body)
        self.assertEqual(b["content"][0]["type"], "text")
        self.assertEqual(b["stop_reason"], "end_turn")

    def test_error_modes(self):
        for mode, status, et in (("throttle", 429, "ThrottlingException"),
                                 ("unavailable", 503, "ServiceUnavailableException"),
                                 ("error", 500, "InternalServerException"),
                                 ("validation", 400, "ValidationException")):
            self.h.set_modes({"chat": {"_default": mode}})
            s, h, b = self.invoke(claude("hi"))
            self.assertEqual((s, h["x-amzn-ErrorType"]), (status, et))
            self.assertIn("message", b)

    def test_truncate(self):
        self.h.set_modes({"chat": {"_default": "truncate"}})
        s, _, b = self.invoke(claude("x" * 100))
        full = "Mock reply to: " + "x" * 100
        self.assertEqual(b["content"][0]["text"], full[: len(full) // 2])
        self.assertEqual(b["stop_reason"], "max_tokens")
        self.assertEqual(b["usage"]["output_tokens"], 100)

    def test_stall(self):
        self.h.set_modes({"chat": {"_default": "stall"}, "_stall_seconds": 0.1})
        t = time.time()
        s, _, b = self.invoke(claude("hi"))
        self.assertGreaterEqual(time.time() - t, 0.1)
        self.assertEqual(s, 200)
        self.assertIn("Mock reply", b["content"][0]["text"])

    def test_modes_reread_and_needle(self):
        self.assertEqual(self.invoke(claude("payroll question"))[0], 200)
        self.h.set_modes({"chat": {"PAYROLL": "throttle"}})
        self.assertEqual(self.invoke(claude("a payroll question"))[0], 429)
        self.assertEqual(self.invoke(claude("something else"))[0], 200)
        self.h.set_modes({})
        self.assertEqual(self.invoke(claude("a payroll question"))[0], 200)

    def test_scenario_overrides_pack(self):
        sc = self.h.dir.parent / "mockbedrock_sc.json"
        self.addCleanup(lambda: sc.unlink())
        sc.write_text(json.dumps({"scenarios": [
            {"id": "a", "match": {"prompt": "magic"}, "reply": {"text": "scripted"}},
            {"id": "b", "match": {"model": "haiku", "prompt": "boom"},
             "reply": {"status": 429, "error_type": "ThrottlingException", "text": "slow"}}]}))
        self.h.stop()
        self.h = Harness(scenarios_file=sc)
        s, _, b = self.invoke(claude("a magic word"))
        self.assertEqual(b["content"][0]["text"], "scripted")
        self.assertEqual(self.invoke(claude("boom"))[0], 429)
        self.assertIn("Mock reply", self.invoke(claude("plain"))[2]["content"][0]["text"])
        self.assertEqual(self.h.requests_log()[0]["mode"], "scenario")

    def test_duplicate_scenario_ids(self):
        sc = self.h.dir / "dup.json"
        sc.write_text(json.dumps({"scenarios": [{"id": "a", "reply": {}}, {"id": "a", "reply": {}}]}))
        with self.assertRaises(ValueError):
            Harness(scenarios_file=sc)

    def test_cohere_embeddings(self):
        texts = ["vacation policy for employees", "employees vacation days policy", "quantum chromodynamics lattice"]
        s, _, b = self.h.request("/model/cohere.embed-english-v3/invoke",
                                 {"texts": texts, "input_type": "search_document"})
        self.assertEqual(s, 200)
        self.assertEqual(b["response_type"], "embeddings_floats")
        self.assertEqual(b["texts"], texts)
        e = b["embeddings"]
        self.assertEqual(len(e), 3)
        self.assertEqual(len(e[0]), 1024)
        for v in e:
            self.assertAlmostEqual(math.sqrt(dot(v, v)), 1.0, places=6)
        self.assertGreater(dot(e[0], e[1]), dot(e[0], e[2]) + 0.3)
        _, _, b2 = self.h.request("/model/cohere.embed-english-v3/invoke", {"texts": texts[:1]})
        self.assertEqual(b2["embeddings"][0], e[0])

    def test_titan_embeddings(self):
        s, _, b = self.h.request("/model/amazon.titan-embed-text-v2:0/invoke", {"inputText": "hello world"})
        self.assertEqual(len(b["embedding"]), 1024)
        self.assertGreaterEqual(b["inputTextTokenCount"], 1)
        _, _, b2 = self.h.request("/model/amazon.titan-embed-text-v2:0/invoke",
                                  {"inputText": "hello world", "dimensions": 256})
        self.assertEqual(len(b2["embedding"]), 256)
        self.assertAlmostEqual(math.sqrt(dot(b2["embedding"], b2["embedding"])), 1.0, places=6)

    def test_embed_dims_option(self):
        self.h.stop()
        self.h = Harness(embed_dims=64)
        _, _, b = self.h.request("/model/cohere.embed-english-v3/invoke", {"texts": ["a b c"]})
        self.assertEqual(len(b["embeddings"][0]), 64)

    def test_requests_log(self):
        self.invoke(claude("log me"))
        self.h.request("/model/cohere.embed-english-v3/invoke", {"texts": ["a", "b"]})
        rows = self.h.requests_log()
        self.assertEqual(len(rows), 2)
        r = rows[0]
        for k in ("at", "api", "path", "model", "family", "mode", "status", "prompt"):
            self.assertIn(k, r)
        self.assertEqual((r["api"], r["model"], r["family"], r["mode"], r["status"]),
                         ("invoke", MODEL, "chat", "answer", 200))
        self.assertIn("log me", r["prompt"])
        self.assertEqual(rows[1]["texts"], 2)
        self.assertNotIn("prompt", rows[1])

    def test_malformed_nested_values_logged_400(self):
        conv_text = {"messages": [{"role": "user", "content": [{"text": 123}]}]}
        conv_id = {"messages": [
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [{"toolUse": {"toolUseId": ["a"], "name": "t", "input": {}}}]},
            {"role": "user", "content": [{"toolResult": {"toolUseId": ["a"], "content": [{"text": "r"}]}}]}]}
        iv = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 50}
        inv_text = {**iv, "messages": [{"role": "user", "content": [{"type": "text", "text": 123}]}]}
        inv_use = {**iv, "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": ["a"], "name": "t", "input": {}}]}]}
        inv_res = {**iv, "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "t", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": ["a"], "content": "r"}]}]}
        cases = [(p, b) for p in ("converse", "converse-stream") for b in (conv_text, conv_id)]
        cases += [(p, b) for p in ("invoke", "invoke-with-response-stream") for b in (inv_text, inv_use, inv_res)]
        for strict in (False, True):
            h = Harness(strict=strict)
            self.addCleanup(h.stop)
            for n, (path, body) in enumerate(cases, 1):
                with self.subTest(strict=strict, path=path, n=n):
                    s, _, b = h.request(f"/model/{MODEL}/{path}", body)
                    self.assertEqual(s, 400, b)
                    self.assertEqual(len(h.requests_log()), n)
                    self.assertEqual(h.requests_log()[-1]["status"], 400)

    def test_arn_model_id(self):
        arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-haiku"
        s, _, b = self.h.request(f"/model/{quote(arn, safe='')}/invoke", claude("hi"))
        self.assertEqual(s, 200)
        self.assertEqual(b["model"], arn)
        self.assertEqual(self.h.requests_log()[-1]["model"], arn)

    def test_bad_json_on_chat_routes(self):
        for op in ("converse", "converse-stream", "invoke-with-response-stream"):
            s, h, _ = self.h.request(f"/model/{MODEL}/{op}", [1])
            self.assertEqual((s, h["x-amzn-ErrorType"]), (400, "ValidationException"))

    def test_bad_json_is_400(self):
        req = urllib.request.Request(f"{self.h.base}/model/{MODEL}/invoke", data=b"{nope", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
