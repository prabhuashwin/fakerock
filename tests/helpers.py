import json
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

from mock_bedrock.server import make_server


class Harness:
    def __init__(self, **kw):
        self.dir = Path(tempfile.mkdtemp(prefix="mockbedrock_"))
        kw.setdefault("log_dir", self.dir)
        self.server = make_server(port=0, **kw)
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self.modes_path = self.dir / "modes.json"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        shutil.rmtree(self.dir, ignore_errors=True)

    def set_modes(self, modes):
        self.modes_path.write_text(json.dumps(modes), encoding="utf-8")

    def request(self, path, body=None, method="POST"):
        """Return (status, headers, parsed-json-or-bytes)."""
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                status, headers, raw = r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            status, headers, raw = e.code, e.headers, e.read()
        try:
            return status, headers, json.loads(raw)
        except ValueError:
            return status, headers, raw

    def requests_log(self):
        p = self.dir / "requests.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
