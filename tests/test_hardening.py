import contextlib
import http.client
import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from mock_bedrock.control import ScenarioError
from mock_bedrock.packs.base import Pack, Reply, load_pack
from mock_bedrock.server import make_server
from tests.helpers import Harness

CLAUDE = "anthropic.claude-3-haiku-20240307-v1:0"
GOOD = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}]}
CONVERSE = {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}


def raw_post(h, path, body):
    c = http.client.HTTPConnection("127.0.0.1", h.port, timeout=10)
    c.request("POST", path, body=body, headers={"content-type": "application/json"})
    r = c.getresponse()
    out = (r.status, r.getheader("x-amzn-ErrorType"), r.read())
    c.close()
    return out


class KeepAliveAndChunking(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.addCleanup(self.h.stop)

    def conn(self):
        c = http.client.HTTPConnection("127.0.0.1", self.h.port, timeout=10)
        self.addCleanup(c.close)
        return c

    def test_stream_is_http11_chunked(self):
        c = self.conn()
        c.request("POST", f"/model/{CLAUDE}/converse-stream", body=json.dumps(CONVERSE))
        r = c.getresponse()
        self.assertEqual(r.version, 11)
        self.assertEqual(r.status, 200)
        self.assertEqual(r.getheader("transfer-encoding"), "chunked")
        self.assertIsNone(r.getheader("content-length"))
        self.assertTrue(r.chunked)
        data = r.read()  # http.client de-chunks and verifies the terminating chunk
        self.assertIn(b"messageStop", data)
        self.assertIn(b"metadata", data)

    def test_keep_alive_two_requests(self):
        c = self.conn()
        socks = set()
        for _ in range(2):
            c.request("POST", f"/model/{CLAUDE}/invoke", body=json.dumps(GOOD))
            socks.add(id(c.sock))
            r = c.getresponse()
            self.assertEqual(r.version, 11)
            self.assertIsNotNone(r.getheader("content-length"))
            self.assertEqual(r.status, 200)
            self.assertEqual(json.loads(r.read())["content"][0]["text"], "Mock reply to: hi")
            self.assertIsNotNone(c.sock)  # server did not ask to close
        self.assertEqual(len(socks), 1)

    def test_keep_alive_across_stream_errors_and_404(self):
        c = self.conn()
        seq = [("POST", f"/model/{CLAUDE}/converse-stream", json.dumps(CONVERSE), 200),
               ("POST", f"/model/{CLAUDE}/invoke", "", 400),
               ("POST", "/nope", json.dumps(GOOD), 404),
               ("GET", "/health", None, 200),
               ("GET", "/nope", None, 404),
               ("POST", f"/model/{CLAUDE}/invoke", json.dumps(GOOD), 200)]
        socks = set()
        for method, path, body, status in seq:
            c.request(method, path, body=body)
            socks.add(id(c.sock))
            r = c.getresponse()
            r.read()
            self.assertIsNotNone(c.sock, path)
            self.assertEqual(r.status, status, path)
        self.assertEqual(len(socks), 1)

    def test_500_has_length_and_keeps_connection(self):
        class Boom(Pack):
            def reply(self, call, family, mode, ctx):
                raise RuntimeError("pack bug")
        h = Harness(pack=Boom())
        self.addCleanup(h.stop)
        c = http.client.HTTPConnection("127.0.0.1", h.port, timeout=10)
        self.addCleanup(c.close)
        with contextlib.redirect_stderr(io.StringIO()):
            for _ in range(2):
                c.request("POST", f"/model/{CLAUDE}/invoke", body=json.dumps(GOOD))
                r = c.getresponse()
                self.assertEqual(r.status, 500)
                self.assertIsNotNone(r.getheader("content-length"))
                r.read()


class Validation(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.addCleanup(self.h.stop)

    def assert_400(self, path, body):
        status, etype, _ = raw_post(self.h, path, body)
        self.assertEqual((status, etype), (400, "ValidationException"), body)

    def test_invoke_inputs(self):
        p = f"/model/{CLAUDE}/invoke"
        for body in (b"", b"[1]", b"\"x\"", b"null", b"{}",
                     json.dumps({"max_tokens": 5}),
                     json.dumps({"messages": "hi"}),
                     json.dumps({**GOOD, "max_tokens": "5"}),
                     json.dumps({**GOOD, "max_tokens": 1.5}),
                     json.dumps({"messages": ["x"]})):
            self.assert_400(p, body)
            self.assert_400(f"/model/{CLAUDE}/invoke-with-response-stream", body)

    def test_converse_inputs(self):
        for op in ("converse", "converse-stream"):
            p = f"/model/{CLAUDE}/{op}"
            for body in (b"", b"[1]", b"{}", json.dumps({"messages": {}}),
                         json.dumps({**CONVERSE, "inferenceConfig": {"maxTokens": "9"}}),
                         json.dumps({**CONVERSE, "inferenceConfig": {"maxTokens": 2.5}}),
                         json.dumps({**CONVERSE, "inferenceConfig": []})):
                self.assert_400(p, body)

    def test_embedding_inputs(self):
        for model, body in (("cohere.embed-english-v3", {}), ("cohere.embed-english-v3", {"texts": "a"}),
                            ("cohere.embed-english-v3", {"texts": [1]}),
                            ("amazon.titan-embed-text-v2:0", {}), ("amazon.titan-embed-text-v1", {"inputText": 3}),
                            ("amazon.titan-embed-text-v2:0", {"inputText": "a", "dimensions": 7})):
            self.assert_400(f"/model/{model}/invoke", json.dumps(body))

    def test_valid_requests_still_ok(self):
        self.assertEqual(raw_post(self.h, f"/model/{CLAUDE}/converse", json.dumps(CONVERSE))[0], 200)
        self.assertEqual(raw_post(self.h, f"/model/{CLAUDE}/converse",
                                  json.dumps({**CONVERSE, "inferenceConfig": {"maxTokens": 5}}))[0], 200)
        self.assertEqual(raw_post(self.h, f"/model/{CLAUDE}/invoke",
                                  json.dumps({"messages": []}))[0], 200)

    def test_output_tokens_capped_at_max_tokens(self):
        class Long(Pack):
            def reply(self, call, family, mode, ctx):
                return Reply(text="x" * 400)
        h = Harness(pack=Long())
        self.addCleanup(h.stop)
        _, _, b = h.request(f"/model/{CLAUDE}/invoke", {**GOOD, "max_tokens": 7})
        self.assertEqual(b["usage"]["output_tokens"], 7)
        _, _, b = h.request(f"/model/{CLAUDE}/converse",
                            {**CONVERSE, "inferenceConfig": {"maxTokens": 9}})
        self.assertEqual(b["usage"]["outputTokens"], 9)
        _, _, b = h.request(f"/model/{CLAUDE}/invoke", {**GOOD, "max_tokens": 5000})
        self.assertEqual(b["usage"]["output_tokens"], 100)


class EmbeddingDims(unittest.TestCase):
    def dims(self, h, model, body):
        s, _, b = h.request(f"/model/{model}/invoke", body)
        self.assertEqual(s, 200)
        return len(b["embedding"] if "embedding" in b else b["embeddings"][0])

    def test_defaults(self):
        h = Harness()
        self.addCleanup(h.stop)
        t = {"inputText": "hello world"}
        self.assertEqual(self.dims(h, "amazon.titan-embed-text-v1", t), 1536)
        self.assertEqual(self.dims(h, "amazon.titan-embed-text-v1", {**t, "dimensions": 256}), 1536)
        self.assertEqual(self.dims(h, "amazon.titan-embed-text-v2:0", t), 1024)
        for d in (256, 512, 1024):
            self.assertEqual(self.dims(h, "amazon.titan-embed-text-v2:0", {**t, "dimensions": d}), d)
        self.assertEqual(self.dims(h, "cohere.embed-english-v3", {"texts": ["a b"]}), 1024)

    def test_override(self):
        h = Harness(embed_dims=64)
        self.addCleanup(h.stop)
        t = {"inputText": "hello world"}
        self.assertEqual(self.dims(h, "amazon.titan-embed-text-v1", t), 64)
        self.assertEqual(self.dims(h, "amazon.titan-embed-text-v2:0", t), 64)
        self.assertEqual(self.dims(h, "cohere.embed-english-v3", {"texts": ["a"]}), 64)

    def test_env_var_default(self):
        import os
        from unittest import mock
        from mock_bedrock import server
        with mock.patch.dict(os.environ, {"MOCK_BEDROCK_EMBED_DIMS": "32"}), \
                mock.patch.object(server, "make_server") as ms:
            ms.return_value.server_address = ("h", 1)
            ms.return_value.serve_forever.side_effect = KeyboardInterrupt
            with contextlib.redirect_stdout(io.StringIO()):
                server.main([])
        self.assertEqual(ms.call_args[0][-1], 32)


class ModesRobustness(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.addCleanup(self.h.stop)
        self.err = io.StringIO()

    def mode_of(self):
        self.h.request(f"/model/{CLAUDE}/invoke", GOOD)
        return self.h.requests_log()[-1]["mode"]

    def test_bad_modes_keep_last_good_and_warn_once(self):
        self.h.set_modes({"chat": {"_default": "throttle"}})
        self.assertEqual(self.mode_of(), "throttle")
        bads = ["{not json", json.dumps({"chat": "throttle"}), json.dumps({"chat": {"_default": 5}}),
                json.dumps([1])]
        for bad in bads:
            self.err = io.StringIO()
            self.h.modes_path.write_text(bad, encoding="utf-8")
            with contextlib.redirect_stderr(self.err):
                for _ in range(3):
                    status, _, _ = self.h.request(f"/model/{CLAUDE}/invoke", GOOD)
                    self.assertEqual(status, 429)  # last good modes still in force
            lines = [l for l in self.err.getvalue().splitlines() if l.strip()]
            self.assertEqual(len(lines), 1, (bad, lines))
            self.assertIn("warning", lines[0])
        self.h.set_modes({"chat": {"_default": "answer"}})
        self.assertEqual(self.mode_of(), "answer")

    def test_bad_modes_at_start_means_defaults(self):
        self.h.modes_path.write_text("{oops", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.mode_of(), "answer")


class ScenarioValidation(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="mb_sc_"))
        self.addCleanup(shutil.rmtree, self.dir, True)

    def make(self, scenarios):
        f = self.dir / "s.json"
        f.write_text(json.dumps({"scenarios": scenarios}), encoding="utf-8")
        s = make_server(port=0, log_dir=self.dir / "log", scenarios_file=f)
        s.server_close()

    def fails(self, scenarios, needle):
        with self.assertRaises(ScenarioError) as cm:
            self.make(scenarios)
        self.assertIn(needle, str(cm.exception))

    def test_invalid(self):
        self.fails([{"id": "a"}, {"id": "a"}], "'a'")
        self.fails([{"id": "rx", "match": {"prompt": "("}}], "'rx'")
        self.fails([{"id": "m", "match": {"model": "[x"}}], "'m'")
        self.fails([{"id": "t1", "reply": {"tool_use": "x"}}], "'t1'")
        self.fails([{"id": "t2", "reply": {"tool_use": {"input": {}}}}], "'t2'")
        self.fails([{"id": "t3", "reply": {"tool_use": {"name": ""}}}], "'t3'")
        self.fails([{"id": "t4", "reply": {"tool_use": {"name": "n", "input": []}}}], "'t4'")
        self.fails([{"id": "st", "reply": {"status": "503"}}], "'st'")

    def test_bad_file(self):
        f = self.dir / "s.json"
        f.write_text("{nope", encoding="utf-8")
        with self.assertRaises(ScenarioError):
            make_server(port=0, log_dir=self.dir / "log", scenarios_file=f)

    def test_valid_and_tool_input_default(self):
        self.make([{"id": "ok", "match": {"prompt": "x"}, "reply": {"tool_use": {"name": "n"}}}])


class LogFailures(unittest.TestCase):
    def test_unwritable_log_dir_falls_back(self):
        d = Path(tempfile.mkdtemp(prefix="mb_log_"))
        self.addCleanup(shutil.rmtree, d, True)
        blocker = d / "file"
        blocker.write_text("x")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            h = Harness(log_dir=blocker / "sub")
        self.addCleanup(h.stop)
        self.assertEqual(err.getvalue().count("warning"), 1)
        self.assertEqual(h.request(f"/model/{CLAUDE}/invoke", GOOD)[0], 200)

    def test_log_write_failure_never_fails_request(self):
        h = Harness()
        self.addCleanup(h.stop)
        shutil.rmtree(h.dir)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            for _ in range(3):
                self.assertEqual(h.request(f"/model/{CLAUDE}/invoke", GOOD)[0], 200)
        self.assertEqual(err.getvalue().count("warning"), 1)
        h.dir.mkdir()  # so cleanup works


class PackLoading(unittest.TestCase):
    def test_unknown_pack_lists_options(self):
        with self.assertRaises(RuntimeError) as cm:
            load_pack("nope")
        self.assertIn("generic", str(cm.exception))
        self.assertIn(".py", str(cm.exception))

    def test_pack_file(self):
        d = Path(tempfile.mkdtemp(prefix="mb_pack_"))
        self.addCleanup(shutil.rmtree, d, True)
        f = d / "p.py"
        f.write_text("from mock_bedrock.packs.base import Pack, Reply\n"
                     "class P(Pack):\n    name='mine'\n"
                     "    def reply(self, call, family, mode, ctx):\n        return Reply(text='custom')\n"
                     "PACK = P\n", encoding="utf-8")
        self.assertEqual(load_pack(str(f)).name, "mine")

    def _write(self, name, body):
        d = Path(tempfile.mkdtemp(prefix="mb_pack_"))
        self.addCleanup(shutil.rmtree, d, True)
        f = d / name
        f.write_text(body, encoding="utf-8")
        return f

    def test_pack_with_dataclass_and_future_annotations(self):
        import sys
        f = self._write("dc.py", """
from __future__ import annotations
from dataclasses import dataclass
from mock_bedrock.packs.base import Pack, Reply

@dataclass
class Cfg:
    word: str
    n: int = 2

class P(Pack):
    name = 'dc'
    def reply(self, call, family, mode, ctx):
        c = Cfg('hi')
        return Reply(text=c.word * c.n)
PACK = P
""")
        pack = load_pack(str(f))
        self.assertEqual(pack.reply(None, "chat", "answer", None).text, "hihi")
        self.assertIn(type(pack).__module__, sys.modules)

    def test_two_packs_do_not_clobber(self):
        import sys
        body = """
from mock_bedrock.packs.base import Pack, Reply
class P(Pack):
    name = '%s'
    def reply(self, call, family, mode, ctx):
        return Reply(text='%s')
PACK = P
"""
        a = load_pack(str(self._write("a.py", body % ("a", "reply-a"))))
        b = load_pack(str(self._write("b.py", body % ("b", "reply-b"))))
        self.assertNotEqual(type(a).__module__, type(b).__module__)
        self.assertIn(type(a).__module__, sys.modules)
        self.assertIn(type(b).__module__, sys.modules)
        self.assertEqual(a.reply(None, "chat", "answer", None).text, "reply-a")
        self.assertEqual(b.reply(None, "chat", "answer", None).text, "reply-b")

    def test_failing_pack_leaves_no_sys_modules_entry(self):
        import sys
        f = self._write("bad.py", "raise ValueError('boom')")
        before = set(sys.modules)
        with self.assertRaises(ValueError):
            load_pack(str(f))
        stale = [m for m in set(sys.modules) - before if m.startswith("mock_bedrock_user_pack")]
        self.assertEqual(stale, [])


if __name__ == "__main__":
    unittest.main()
