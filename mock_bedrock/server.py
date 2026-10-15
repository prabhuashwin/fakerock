"""HTTP server: routes, common modes, error shaping, request logging.

Supported APIs: InvokeModel, InvokeModelWithResponseStream, Converse, ConverseStream.
Common modes: answer, stall, truncate, stream_error, drop, plus the error modes in control.ERROR_MODES:
throttle, unavailable, error, validation, model_timeout, model_not_ready, access_denied, not_found,
quota_exceeded, model_error.
(stream_error: streaming APIs emit a first delta then a modelStreamErrorException; others act like error.
drop: non-streaming APIs close the connection with no response; streaming APIs send the first event, then close.)
modes.json also accepts _stall_seconds, _latency_seconds (delay before every reply) and
_chunk_delay_seconds (delay between streamed events).
"""
import argparse
import json
import os
import re
import socket
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from mock_bedrock import __version__
from mock_bedrock.control import ERROR_MODES, Control, ScenarioError
from mock_bedrock.ids import IdSource
from mock_bedrock.embeddings import handle_embedding, is_embedding_model
from mock_bedrock.converse import build_converse_response, converse_stream_events, parse_converse
from mock_bedrock.invoke import build_invoke_response, invoke_stream_events, parse_invoke
from mock_bedrock.packs.base import Call, Reply, load_pack

ROUTE_RE = re.compile(r"^/model/(.+)/(invoke|invoke-with-response-stream|converse|converse-stream)$")

class HttpResult:
    def __init__(self, status, body: bytes, error_type=None):
        self.status, self.body, self.error_type = status, body, error_type
        self.drop = False  # close the connection instead of (fully) responding
        self.events = None  # list of encoded event-stream messages -> streamed response
        self.request_id = None  # sent as x-amzn-RequestId (new uuid when None)


def error_result(status, error_type, message):
    return HttpResult(status, json.dumps({"message": message}).encode(), error_type)


def reply_to_result(reply: Reply, ok_builder):
    if reply.raw_body is not None:
        return HttpResult(reply.status, reply.raw_body, reply.error_type)
    if reply.status >= 400 or reply.error_type:
        return error_result(reply.status if reply.status >= 400 else 500,
                            reply.error_type or "InternalServerException",
                            reply.message or "mock error")
    result = HttpResult(reply.status, json.dumps(ok_builder()).encode())
    result.drop = reply.drop
    return result


def compute_reply(app, call, family, mode):
    """Apply the common modes; everything else goes to the pack. Returns a Reply."""
    if mode == "drop":
        r = app.pack.reply(call, family, "answer", app.control.ctx)
        r.drop = True
        return r
    if mode in ERROR_MODES:
        status, et, msg = ERROR_MODES[mode]
        return Reply(status=status, error_type=et, message=msg)
    if mode == "stream_error":  # only meaningful for streaming APIs; elsewhere it behaves like "error"
        return Reply(status=500, error_type="InternalServerException", message="Internal server error (mock)")
    if mode in ("stall", "truncate"):
        if mode == "stall":
            time.sleep(app.control.stall_seconds())
        r = app.pack.reply(call, family, "answer", app.control.ctx)
        if mode == "truncate":
            r.text = r.text[: len(r.text) // 2]
            r.stop_reason, r.output_tokens = "max_tokens", call.max_tokens
        return r
    return app.pack.reply(call, family, mode, app.control.ctx)


def _latency(app):
    delay = app.control.latency_seconds()
    if delay > 0:
        time.sleep(delay)


def resolve(app, call, family, streaming):
    """One decision point for all chat-like APIs.
    Returns (mode_label, reply, stream_error, scenario_id)."""
    _latency(app)
    sel = app.control.select_scenario(call)
    if sel is not None:
        sid, spec = sel
        mode = spec.get("mode")
        if mode is None:
            return "scenario", app.control.reply_from_spec(spec), False, sid
        label = "scenario"
    else:
        sid = None
        mode = app.control.mode_for(app.pack, call, family)
        label = mode
    stream_error = False
    if mode == "stream_error" and streaming:
        stream_error, mode = True, "answer"
    return label, compute_reply(app, call, family, mode), stream_error, sid


def _embedding_text(body):
    """Input text(s) of an embedding request, for mode needles and scenario matching."""
    texts = body.get("texts")
    if isinstance(texts, list) and all(isinstance(t, str) for t in texts):
        return "\n".join(texts)
    text = body.get("inputText")
    return text if isinstance(text, str) else ""


def resolve_embedding(app, call):
    """Modes and scenarios for embedding models. Returns (mode_label, error_reply_or_None, scenario_id);
    a None reply means a normal embedding response."""
    _latency(app)
    sel = app.control.select_scenario(call)
    if sel is not None:
        sid, spec = sel
        mode = spec.get("mode")
        if mode is None:
            reply = app.control.reply_from_spec(spec)
            failed = reply.raw_body is not None or reply.status >= 400 or reply.error_type
            return "scenario", (reply if failed else None), sid
        label = "scenario"
    else:
        sid = None
        mode = app.control.mode_for(app.pack, call, "embedding")
        label = mode
    if mode == "drop":
        return label, Reply(drop=True), sid
    if mode in ERROR_MODES:
        status, et, msg = ERROR_MODES[mode]
        return label, Reply(status=status, error_type=et, message=msg), sid
    if mode == "stream_error":
        return label, Reply(status=500, error_type="InternalServerException",
                            message="Internal server error (mock)"), sid
    if mode == "stall":
        time.sleep(app.control.stall_seconds())
    return label, None, sid


def _parse_json_object(body_bytes):
    if not body_bytes or not body_bytes.strip():
        raise ValueError("empty request body")
    body = json.loads(body_bytes)
    if not isinstance(body, dict):
        raise ValueError("body must be a JSON object")
    return body


def _capture(body_bytes):
    """Fields describing the request body: parsed object, or null plus truncated raw text."""
    try:
        return {"body": _parse_json_object(body_bytes)}
    except ValueError:
        return {"body": None, "raw": (body_bytes or b"").decode("utf-8", errors="replace")[:2000]}


def _row(api, path, model, rid, stream, cap, **fields):
    return {"at": time.time(), "api": api, "path": path, "model": model, "request_id": rid,
            "stream": stream, **cap, "scenario": None, **fields}


def handle_invoke(app, h, model, body_bytes, path):
    rid, cap = app.ids.request_id(), _capture(body_bytes)

    def done(result, **fields):
        result.request_id = rid
        app.control.log_request(_row("invoke", path, model, rid, False, cap, **fields))
        return result
    try:
        body = _parse_json_object(body_bytes)
        if is_embedding_model(model):
            resp, n = handle_embedding(model, body, app.embed_dims, app.ids.embed_id)
        else:
            call = parse_invoke(model, body)
    except ValueError as e:
        return done(error_result(400, "ValidationException", f"Malformed input request: {e}"),
                    family=None, mode=None, status=400, prompt="")
    if is_embedding_model(model):
        call = Call(api="invoke", model=model, messages=[{"role": "user", "text": _embedding_text(body)}])
        label, reply, sid = resolve_embedding(app, call)
        result = reply_to_result(reply, lambda: resp) if reply is not None else HttpResult(200, json.dumps(resp).encode())
        return done(result, family="embedding", mode=label, status=0 if result.drop else result.status, texts=n,
                    scenario=sid if label == "scenario" else None)
    family = app.pack.family(call)
    mode, reply, _, sid = resolve(app, call, family, False)
    result = reply_to_result(reply, lambda: build_invoke_response(call, reply, app.ids.msg_id()))
    return done(result, family=family, mode=mode, status=0 if result.drop else result.status, prompt=call.prompt,
                scenario=sid if mode == "scenario" else None)


def _make_chat_handler(api, parse, build_response, stream_events=None):
    """Build a handler for converse / converse-stream / invoke-with-response-stream."""
    streaming = stream_events is not None

    def handler(app, h, model, body_bytes, path):
        rid, cap = app.ids.request_id(), _capture(body_bytes)
        kw = (lambda: {"msg_id": app.ids.msg_id()}) if api.startswith("invoke") else dict

        def done(result, family=None, mode=None, prompt="", scenario=None):
            result.request_id = rid
            app.control.log_request(_row(api, path, model, rid, streaming, cap, family=family, mode=mode,
                                         status=0 if result.drop else result.status, prompt=prompt,
                                         scenario=scenario))
            return result
        try:
            body = _parse_json_object(body_bytes)
            call = parse(model, body, streaming)
        except ValueError as e:
            return done(error_result(400, "ValidationException", f"Malformed input request: {e}"))
        family = app.pack.family(call)
        mode, reply, stream_error, sid = resolve(app, call, family, streaming)
        failed = reply.raw_body is not None or reply.status >= 400 or reply.error_type
        if not streaming or failed:
            result = reply_to_result(reply, lambda: build_response(call, reply, **kw()))
        else:
            result = HttpResult(200, b"")
            result.events = stream_events(call, reply, stream_error, **kw())
            result.drop = reply.drop
        return done(result, family, mode, call.prompt, sid if mode == "scenario" else None)
    return handler


# operation -> handler(app, handler, model, body_bytes, path) -> HttpResult.
ROUTES = {
    "invoke": handle_invoke,
    "invoke-with-response-stream": _make_chat_handler(
        "invoke-stream", parse_invoke, build_invoke_response, invoke_stream_events),
    "converse": _make_chat_handler("converse", parse_converse, build_converse_response),
    "converse-stream": _make_chat_handler(
        "converse-stream", parse_converse, build_converse_response, converse_stream_events),
}


class Handler(BaseHTTPRequestHandler):
    app = None
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass  # client dropped a kept-alive connection

    def _hang_up(self):
        self.close_connection = True
        try:
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def _send(self, result: HttpResult):
        if result.drop and result.events is None:
            self._hang_up()
            return
        if result.events is not None:
            # Streaming: one HTTP chunk per event-stream message, then the terminating chunk.
            self.send_response(200)
            self.send_header("content-type", "application/vnd.amazon.eventstream")
            self.send_header("x-amzn-RequestId", result.request_id or self.app.ids.anon_id())
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            self.wfile.flush()
            delay = self.app.control.chunk_delay_seconds()
            for i, ev in enumerate(result.events):
                if i and delay > 0:
                    time.sleep(delay)
                self.wfile.write(b"%x\r\n" % len(ev) + ev + b"\r\n")
                self.wfile.flush()
                if result.drop:  # first event sent; hang up without the terminating chunk
                    self._hang_up()
                    return
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        self.send_response(result.status)
        self.send_header("content-type", "application/json")
        self.send_header("x-amzn-RequestId", result.request_id or self.app.ids.anon_id())
        if result.error_type:
            self.send_header("x-amzn-ErrorType", result.error_type)
        self.send_header("content-length", str(len(result.body)))
        self.end_headers()
        self.wfile.write(result.body)

    def _read_body(self) -> bytes:
        """Always consume the whole request body so keep-alive connections stay in sync."""
        if "chunked" in self.headers.get("transfer-encoding", "").lower():
            out = bytearray()
            while True:
                size = int(self.rfile.readline(65537).split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline(65537).strip():  # trailers
                        pass
                    return bytes(out)
                out += self.rfile.read(size)
                self.rfile.readline(65537)
        length = int(self.headers.get("content-length", 0) or 0)
        return self.rfile.read(length) if length > 0 else b""

    def _control(self, method, path, body):
        """Handle /_fakerock/* routes; these are never written to the request log."""
        control = self.app.control
        allowed = {"/_fakerock/requests": ("GET",), "/_fakerock/scenarios": ("PUT", "POST"),
                   "/_fakerock/modes": ("PUT", "POST"), "/_fakerock/reset": ("POST",)}
        if path not in allowed:
            return error_result(404, "UnknownOperationException", f"unknown path {path}")
        if method not in allowed[path]:
            return error_result(405, "MethodNotAllowed", f"{method} not allowed on {path}")
        if path == "/_fakerock/requests":
            return HttpResult(200, json.dumps({"requests": control.requests()}).encode())
        if path == "/_fakerock/reset":
            control.reset()
            return HttpResult(200, b'{"ok": true}')
        try:
            data = json.loads(body) if body and body.strip() else None
        except ValueError as e:
            return error_result(400, "ValidationException", f"invalid JSON: {e}")
        try:
            if path == "/_fakerock/scenarios":
                n = control.set_scenarios(data)
                return HttpResult(200, json.dumps({"ok": True, "count": n}).encode())
            control.set_modes(data)
            return HttpResult(200, b'{"ok": true}')
        except ValueError as e:  # ScenarioError is a ValueError
            return error_result(400, "ValidationException", str(e))

    def do_GET(self):
        try:
            if self.headers.get("content-length") or self.headers.get("transfer-encoding"):
                body = self._read_body()
            else:
                body = b""
            path = self.path.split("?", 1)[0]
            if path.startswith("/_fakerock/"):
                result = self._control("GET", path, body)
            elif path == "/health":
                result = HttpResult(200, json.dumps({"ok": True, "pack": self.app.pack.name}).encode())
            else:
                result = error_result(404, "UnknownOperationException", f"unknown path {self.path}")
        except Exception:
            self.close_connection = True
            result = error_result(400, "ValidationException", "bad request")
        self._finish(result)

    def do_POST(self):
        try:
            try:
                body = self._read_body()
            except (ValueError, OSError):
                self.close_connection = True  # framing is unknown; do not reuse the connection
                self._finish(error_result(400, "ValidationException", "Malformed request body"))
                return
            path = self.path.split("?", 1)[0]
            m = ROUTE_RE.match(path)
            if path.startswith("/_fakerock/"):
                result = self._control(self.command, path, body)
            elif not m:
                result = error_result(404, "UnknownOperationException", f"unknown path {path}")
            else:
                result = ROUTES[m.group(2)](self.app, self, unquote(m.group(1)), body, path)
        except Exception:
            traceback.print_exc()
            result = error_result(500, "InternalServerException", "mock internal error")
        self._finish(result)

    def _finish(self, result):
        try:
            self._send(result)
        except Exception:
            self.close_connection = True  # client went away mid-response

    do_PUT = do_POST


class App:
    def __init__(self, pack, control, embed_dims, ids=None):
        self.pack, self.control, self.embed_dims, self.ids = pack, control, embed_dims, ids or IdSource()


def make_server(host="127.0.0.1", port=52999, pack="generic", pack_options=None, log_dir=None,
                modes_file=None, scenarios_file=None, embed_dims=None, modes=None, scenarios=None, deterministic=False):
    log_dir = Path(log_dir or "./.mock-bedrock").resolve()
    control = Control(log_dir, modes_file, scenarios_file, modes=modes, scenarios=scenarios)
    pack_obj = load_pack(pack, pack_options or {}) if isinstance(pack, str) else pack
    ids = IdSource(deterministic)
    control._reset_hooks.append(ids.reset)
    app = App(pack_obj, control, embed_dims, ids)
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.app = app
    return server


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mock_bedrock", description="Mock AWS Bedrock runtime (never contacts AWS).")
    ap.add_argument("--host", default=os.environ.get("MOCK_BEDROCK_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MOCK_BEDROCK_PORT", "52999")))
    ap.add_argument("--pack", default=os.environ.get("MOCK_BEDROCK_PACK", "generic"),
                    help="generic | path/to/pack.py")
    ap.add_argument("--pack-opt", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--log-dir", default=os.environ.get("MOCK_BEDROCK_LOG_DIR", "./.mock-bedrock"))
    ap.add_argument("--modes", help="modes.json path (default: <log-dir>/modes.json)")
    ap.add_argument("--scenarios", help="scenarios.json path")
    ap.add_argument("--embed-dims", type=int, default=int(os.environ.get("MOCK_BEDROCK_EMBED_DIMS", "0")) or None,
                    help="override embedding dimensions for all models "
                         "(default per model; env MOCK_BEDROCK_EMBED_DIMS)")
    ap.add_argument("--deterministic", action="store_true",
                    default=os.environ.get("MOCK_BEDROCK_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes", "on"),
                    help="counter-based request/message ids instead of random "
                         "(env MOCK_BEDROCK_DETERMINISTIC)")
    ap.add_argument("--version", action="version", version=f"mock_bedrock {__version__}")
    a = ap.parse_args(argv)
    opts = {}
    for kv in a.pack_opt:
        k, _, v = kv.partition("=")
        opts[k] = v
    try:
        server = make_server(a.host, a.port, a.pack, opts, a.log_dir, a.modes, a.scenarios, a.embed_dims,
                             deterministic=a.deterministic)
    except (ScenarioError, RuntimeError) as e:
        raise SystemExit(f"mock_bedrock: {e}")
    host, port = server.server_address[:2]
    print(f"mock bedrock listening on http://{host}:{port} pack={server.app.pack.name} "
          f"log_dir={server.app.control.log_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
