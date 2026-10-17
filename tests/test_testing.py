import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from mock_bedrock.control import ScenarioError
from mock_bedrock.testing import MockBedrock

ROOT = Path(__file__).resolve().parent.parent
BODY = {"max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}


def post(mock, body=BODY):
    req = urllib.request.Request(mock.endpoint_url + "/model/anthropic.x/invoke", method="POST",
                                 data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def accepts(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


class TestMockBedrock(unittest.TestCase):
    def test_pipelined_requests_both_answered(self):
        m = MockBedrock().start()
        try:
            req = b"GET /health HTTP/1.1" + bytes([13,10]) + b"Host: x" + bytes([13,10,13,10])
            with socket.create_connection(("127.0.0.1", m.port), timeout=3) as s:
                s.sendall(req * 2)
                data = b""
                while data.count(b"HTTP/1.1 200") < 2:
                    chunk = s.recv(65536)
                    self.assertTrue(chunk, "connection closed before both responses arrived")
                    data += chunk
        finally:
            m.stop()

    def test_body_split_across_packets(self):
        m = MockBedrock().start()
        try:
            body = json.dumps({"messages": [{"role": "user", "content": [{"text": "x" * 5000}]}]}).encode()
            crlf = "\r\n"
            head = crlf.join(["POST /model/anthropic.claude-test/converse HTTP/1.1", "Host: x",
                              "Content-Type: application/json", "Content-Length: %d" % len(body), "", ""]).encode()
            with socket.create_connection(("127.0.0.1", m.port), timeout=5) as s:
                s.sendall(head + body[:100])
                time.sleep(0.3)
                s.sendall(body[100:])
                self.assertTrue(s.recv(200).startswith(b"HTTP/1.1 200"))
        finally:
            m.stop()

    def test_start_stop_health(self):
        m = MockBedrock().start()
        try:
            with urllib.request.urlopen(m.endpoint_url + "/health", timeout=10) as r:
                self.assertEqual(r.status, 200)
        finally:
            m.stop()

    def test_modes_dict(self):
        with MockBedrock(modes={"chat": {"_default": "throttle"}}) as m:
            self.assertEqual(post(m)[0], 429)

    def test_bad_modes(self):
        with self.assertRaises(ValueError):
            MockBedrock(modes={"chat": "throttle"}).start()

    def test_scenarios_list(self):
        sc = [{"id": "a", "match": {"prompt": "hi"}, "reply": {"text": "scripted"}}]
        with MockBedrock(scenarios=sc) as m:
            status, body = post(m)
            self.assertEqual(status, 200)
            self.assertIn("scripted", json.dumps(body))

    def test_scenarios_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            p.write_text(json.dumps({"scenarios": [{"id": "a", "reply": {"status": 503, "text": "down"}}]}))
            with MockBedrock(scenarios=p) as m:
                self.assertEqual(post(m)[0], 503)

    def test_invalid_scenarios(self):
        with self.assertRaises(ScenarioError):
            MockBedrock(scenarios=[{"id": "bad", "match": {"prompt": "("}}]).start()

    def test_context_manager_cleans_up(self):
        with MockBedrock() as m:
            port, d = m.port, m.log_dir
            self.assertTrue(d.exists())
            self.assertTrue(accepts(port))
        self.assertFalse(accepts(port))
        self.assertFalse(d.exists())

    def test_stop_twice_and_given_log_dir_kept(self):
        with tempfile.TemporaryDirectory() as d:
            m = MockBedrock(log_dir=d).start()
            m.stop()
            m.stop()
            self.assertTrue(Path(d).exists())

    def test_client(self):
        try:
            import boto3  # noqa: F401
        except ImportError:
            self.skipTest("boto3 not installed")
        with MockBedrock() as m:
            r = m.client().invoke_model(modelId="anthropic.x", body=json.dumps(BODY))
            self.assertIn("content", json.loads(r["body"].read()))


class TestPytestPlugin(unittest.TestCase):
    def test_fixture(self):
        try:
            import pytest  # noqa: F401
        except ImportError:
            self.skipTest("pytest not installed")
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "conftest.py").write_text('pytest_plugins = ["mock_bedrock.pytest_plugin"]\n')
            (Path(d) / "test_x.py").write_text(
                "import urllib.request\n"
                "def test_health(fakerock):\n"
                "    with urllib.request.urlopen(fakerock.endpoint_url + '/health') as r:\n"
                "        assert r.status == 200\n")
            env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", d],
                               cwd=d, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)


class TestStopClosesConnections(unittest.TestCase):
    def _client(self, mock):
        try:
            from botocore.config import Config
        except ImportError:
            self.skipTest("boto3 not installed")
        return mock.client(config=Config(retries={"max_attempts": 1}, connect_timeout=2, read_timeout=3))

    def test_keepalive_client_fails_after_stop(self):
        m = MockBedrock().start()
        try:
            br = self._client(m)
            kw = dict(modelId="anthropic.x", messages=[{"role": "user", "content": [{"text": "hi"}]}])
            r = br.converse(**kw)
            self.assertEqual(r["ResponseMetadata"]["HTTPStatusCode"], 200)
            m.stop()
            with self.assertRaises(Exception):
                br.converse(**kw)
        finally:
            m.stop()

    def test_stop_is_fast_with_idle_keepalive(self):
        import time
        m = MockBedrock().start()
        try:
            br = self._client(m)
            br.converse(modelId="anthropic.x", messages=[{"role": "user", "content": [{"text": "hi"}]}])
            t0 = time.monotonic()
            m.stop()
            self.assertLess(time.monotonic() - t0, 1.5)
        finally:
            m.stop()

    def test_stop_interrupts_stalled_request(self):
        import threading
        import time
        errors = []
        old_hook = threading.excepthook
        threading.excepthook = lambda a: errors.append(a)
        err = io.StringIO()
        old_err, sys.stderr = sys.stderr, err
        m = MockBedrock(modes={"chat": {"_default": "stall"}, "_stall_seconds": 30}).start()
        log_dir = m.log_dir
        try:
            def call():
                try:
                    post(m)
                except Exception:
                    pass
            t = threading.Thread(target=call, daemon=True)
            t.start()
            time.sleep(0.5)  # request is now sleeping inside the stall
            t0 = time.monotonic()
            m.stop()
            self.assertLess(time.monotonic() - t0, 1.5)
            self.assertFalse(log_dir.exists())
            t.join(5)
            self.assertFalse(t.is_alive())
            time.sleep(0.5)
            self.assertEqual(errors, [])
            self.assertEqual(err.getvalue(), "")
            self.assertFalse(log_dir.exists())
        finally:
            threading.excepthook = old_hook
            sys.stderr = old_err
            m.stop()

if __name__ == "__main__":
    unittest.main()
