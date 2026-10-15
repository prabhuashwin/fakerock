"""Run the mock inside a test.

    from mock_bedrock.testing import MockBedrock
    with MockBedrock(modes={"chat": {"_default": "throttle"}}) as mock:
        br = mock.client()          # boto3 bedrock-runtime client
        ...                         # or point anything at mock.endpoint_url
        mock.set_modes({"chat": {"_default": "answer"}})   # change behaviour mid-test
        assert mock.requests(api="converse")                # what the mock received
        mock.reset()                # back to startup scenarios, modes.json; clears requests
"""
import json
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

from mock_bedrock.control import load_scenarios
from mock_bedrock.server import make_server

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route to a proxy


class FakerockClient:
    """Control-API client (urllib only) for a running fakerock server."""

    def __init__(self, base_url):
        self.base_url = base_url.rstrip("/")

    def _call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"content-type": "application/json"})
        try:
            with _opener.open(req, timeout=10) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                msg = json.loads(raw).get("message", raw.decode("utf-8", "replace"))
            except (ValueError, AttributeError):
                msg = raw.decode("utf-8", "replace")
            raise RuntimeError(f"fakerock {method} {path} failed ({e.code}): {msg}") from None

    def set_scenarios(self, scenarios):
        return self._call("PUT", "/_fakerock/scenarios", scenarios)

    def set_modes(self, modes):
        return self._call("PUT", "/_fakerock/modes", modes)

    def reset(self):
        return self._call("POST", "/_fakerock/reset")

    def requests(self, api=None, model=None, scenario=None):
        """Logged requests, optionally filtered by api, model, scenario id.

        Row fields: seq, at, api, path, model, request_id (== x-amzn-RequestId), stream, body
        (parsed request JSON, or None with raw= text), scenario, family, mode, status, prompt,
        texts (embeddings only).
        """
        rows = self._call("GET", "/_fakerock/requests")["requests"]
        return [r for r in rows
                if (api is None or r.get("api") == api) and (model is None or r.get("model") == model)
                and (scenario is None or r.get("scenario") == scenario)]

    def last_request(self):
        rows = self.requests()
        return rows[-1] if rows else None


class MockBedrock:
    def __init__(self, *, port=0, host="127.0.0.1", pack="generic", pack_options=None, modes=None,
                 scenarios=None, log_dir=None, embed_dims=None,
                 deterministic=False):
        if isinstance(scenarios, (str, Path)):
            scenarios = load_scenarios(scenarios)
        self._kw = dict(host=host, port=port, pack=pack, pack_options=pack_options, modes=modes,
                        scenarios=scenarios, embed_dims=embed_dims,
                        deterministic=deterministic)
        self._own_dir = log_dir is None
        self._log_dir = Path(log_dir) if log_dir else None
        self._server = self._thread = None

    def start(self):
        if self._server:
            return self
        if self._own_dir:
            self._log_dir = Path(tempfile.mkdtemp(prefix="fakerock_"))
        try:
            self._server = make_server(log_dir=self._log_dir, **self._kw)
        except BaseException:
            self._cleanup()
            raise
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._ctl = FakerockClient(self.endpoint_url)
        return self

    def _cleanup(self):
        if self._own_dir and self._log_dir:
            shutil.rmtree(self._log_dir, ignore_errors=True)

    def stop(self):
        if not self._server:
            return
        server, self._server = self._server, None
        server.shutdown()
        server.server_close()
        self._thread.join(5)
        self._cleanup()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    @property
    def port(self):
        return self._server.server_address[1]

    @property
    def endpoint_url(self):
        host = self._kw["host"]
        return f"http://{'127.0.0.1' if host in ('', '0.0.0.0') else host}:{self.port}"

    @property
    def log_dir(self):
        return self._log_dir

    def set_scenarios(self, scenarios):
        return self._ctl.set_scenarios(scenarios)

    def set_modes(self, modes):
        return self._ctl.set_modes(modes)

    def reset(self):
        return self._ctl.reset()

    def requests(self, api=None, model=None, scenario=None):
        """See FakerockClient.requests for row fields."""
        return self._ctl.requests(api, model, scenario)

    def last_request(self):
        return self._ctl.last_request()

    def client(self, **kw):
        """boto3 bedrock-runtime client pointed at this server."""
        try:
            import boto3
        except ImportError:
            raise ImportError("MockBedrock.client() needs boto3: pip install boto3") from None
        args = dict(region_name="us-east-1", endpoint_url=self.endpoint_url,
                    aws_access_key_id="dummy", aws_secret_access_key="dummy")
        return boto3.client("bedrock-runtime", **{**args, **kw})
