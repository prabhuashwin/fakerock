# fakerock: a local mock server for AWS Bedrock Runtime

[![tests](https://github.com/prabhuashwin/fakerock/actions/workflows/tests.yml/badge.svg)](https://github.com/prabhuashwin/fakerock/actions/workflows/tests.yml)

A local, stdlib-only Python stand-in for the **Amazon Bedrock Runtime** API (AWS Bedrock). Use it to
run an app or its tests against Bedrock-shaped responses. It never contacts AWS, never uses real
credentials, and costs nothing.

> **Unofficial.** This project is not affiliated with, endorsed by, or supported by Amazon Web
> Services. "AWS", "Amazon Bedrock" and related names are trademarks of Amazon.com, Inc. or its
> affiliates and are used here only to describe what the mock is compatible with.

Why use it:

- **No cost, no AWS account.** Develop and run CI without Bedrock calls or credentials.
- **Repeatable.** The same request gets the same reply, so tests don't flake on model output.
- **Failures on demand.** Throttling, outages, stalls, truncated or broken streams, and
  validation errors can be switched on without restarting, so you can test retry and
  error handling.
- **Drop-in.** Real AWS SDKs (boto3 is tested) work unchanged; you only override the
  endpoint URL. Python 3 standard library only, so it also ships as a tiny Docker image.
- **Your own replies.** A small "pack" file decides what the mock answers for your app.

## What's in this repo

| Path | Contents |
|---|---|
| `mock_bedrock/` | The server: routing and modes (`server.py`), InvokeModel (`invoke.py`), Converse (`converse.py`), AWS event-stream encoding (`eventstream.py`), embeddings (`embeddings.py`), live controls and request log (`control.py`), in-test server (`testing.py`), pytest fixture (`pytest_plugin.py`) |
| `mock_bedrock/packs/` | Reply packs: the pack interface (`base.py`) and the default echo pack (`generic.py`) |
| `tests/` | Unit, streaming, hardening and boto3 contract tests |
| `Dockerfile`, `docker-compose.example.yml` | Container image and an example compose service (published on loopback only) |

## Supported APIs

- `InvokeModel`, `InvokeModelWithResponseStream` (Anthropic message bodies)
- `Converse`, `ConverseStream`
- Embeddings via InvokeModel for `cohere.embed*` and `amazon.titan-embed*` model IDs
- `GET /health` -> `{"ok": true, "pack": "<name>"}`

Anything else returns 404 `UnknownOperationException`.

## Quick start

```
python -m mock_bedrock                       # http://127.0.0.1:52999
python -m mock_bedrock --scenarios scenarios.json --port 8080
```

Flags (env var supplying the default in brackets): `--host` [`MOCK_BEDROCK_HOST`, 127.0.0.1],
`--port` [`MOCK_BEDROCK_PORT`, 52999], `--pack` [`MOCK_BEDROCK_PACK`, generic],
`--pack-opt KEY=VALUE` (repeatable), `--log-dir` [`MOCK_BEDROCK_LOG_DIR`, `./.mock-bedrock`],
`--modes` (default `<log-dir>/modes.json`), `--scenarios`, `--embed-dims` [`MOCK_BEDROCK_EMBED_DIMS`,
per-model default], `--deterministic` [`MOCK_BEDROCK_DETERMINISTIC`=1/true], `--version`.

Deterministic ids (also `MockBedrock(deterministic=True)`): instead of random uuids, ids come from per-server
counters, so the same request sequence gives identical output. Request ids (`x-amzn-RequestId` and the log's
`request_id`) are `00000000-0000-4000-8000-000000000001`, `...0002`, ...; Anthropic message ids are
`msg_mock_0000000000000001`, ...; Cohere embed `id` uses its own counter in the request-id format.
`POST /_fakerock/reset` restarts all counters. Log `at` timestamps stay wall-clock.

Docker:

```
docker build -t mock-bedrock .
docker run --rm -p 127.0.0.1:52999:52999 -v "$PWD/.mock-bedrock:/data" mock-bedrock
docker run ... mock-bedrock --scenarios /data/scenarios.json
```

`docker-compose.example.yml` shows a loopback-only publish and a sibling container. The image binds `0.0.0.0` inside
the container only; do not publish the port on a non-loopback host interface. The control routes
(`/_fakerock/*`) let anyone who can reach the port change replies, so keep it on loopback. In Docker change
the port with `-e MOCK_BEDROCK_PORT=...`, not `--port`: the healthcheck reads that variable.
If `/data` is not writable (e.g. a bind mount owned by another uid) the server warns once on
stderr and logs to a temp dir; `modes.json` is still read from `/data`. A failed log write never
fails a request.

## Pointing an app at it

Set `AWS_ENDPOINT_URL_BEDROCK_RUNTIME=http://127.0.0.1:52999` (recent AWS SDKs), or set the
endpoint per client (`endpoint_url` in boto3, `endpoint` in the JS SDK v3, `endpointOverride` in the Java SDK v2).
Use dummy credentials (`AWS_ACCESS_KEY_ID=dummy`, `AWS_SECRET_ACCESS_KEY=dummy`) and a region.
If the override is missed, the request goes to real AWS, fails authentication, and costs nothing.

```python
import boto3, json
br = boto3.client("bedrock-runtime", region_name="us-east-1",
                  endpoint_url="http://127.0.0.1:52999",
                  aws_access_key_id="dummy", aws_secret_access_key="dummy")
r = br.invoke_model(modelId="anthropic.claude-test", body=json.dumps({
    "anthropic_version": "bedrock-2023-05-31", "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}]}))
print(json.loads(r["body"].read()))
```

## Using it from tests

```python
from mock_bedrock.testing import MockBedrock

with MockBedrock(modes={"chat": {"_default": "throttle"}}) as mock:   # random free port
    br = mock.client()            # boto3 client; or use mock.endpoint_url
```

Per-test control (also on `FakerockClient(base_url)` for a server started elsewhere):

```python
mock.set_scenarios([{"id": "t", "match": {"prompt": "hi"}, "reply": {"text": "yo"}}])  # replaces active list
mock.set_modes({"chat": {"_default": "throttle"}})   # overrides modes.json / startup modes
mock.requests(api="converse", model=None)            # captured rows, optionally filtered
mock.last_request()                                  # newest row or None
mock.reset()                                         # startup scenarios/modes, clears requests
```

Server errors raise `RuntimeError` with the server's message.

`modes` is a dict (fixed for the server's life unless `set_modes` is called); `scenarios` is a list, a dict with a `scenarios`
list, or a path to a scenarios file. Bad values raise at start. Logs go to a temp dir
(`mock.log_dir`) that is removed on stop unless you pass `log_dir`.

pytest: add `pytest_plugins = ["mock_bedrock.pytest_plugin"]` to `conftest.py`, then take the
`fakerock` fixture (a started `MockBedrock`, one per test, xdist-safe):

```python
def test_reply(fakerock):
    r = fakerock.client().converse(modelId="anthropic.claude-test",
                                   messages=[{"role": "user", "content": [{"text": "hi"}]}])
    assert r["output"]["message"]["content"][0]["text"] == "Mock reply to: hi"
```

## Controlling replies

The pack classifies each request into a **family** (generic: `chat`, or `tools` when the
request offers tools), or `embedding` for embedding models. The mode comes from `modes.json` in the log dir (or `--modes`),
**re-read on every call**, so you can edit it while the server runs (write it atomically: temp
file, then rename, since a reader can catch a half-written or empty file):

```json
{
  "_stall_seconds": 60,
  "chat":  {"_default": "answer", "summarise": "throttle"},
  "tools": {"_default": "answer"}
}
```

Within a family, each key other than `_default` is a case-insensitive substring tested against
the pack's mode key (generic: last user message text); first match wins, else `_default`, else
the pack's default (`answer`). For `embedding` the mode key is the Cohere `texts` joined with
`"
"`, or the Titan `inputText`. Packs may add their own families and modes.

Common modes (every pack):
- `answer`: normal pack reply
- `stall`: sleep `_stall_seconds` (default 60), then answer
- `truncate`: half the text, `stop_reason` `max_tokens`
- `throttle`: 429 ThrottlingException
- `unavailable`: 503 ServiceUnavailableException
- `error`: 500 InternalServerException
- `validation`: 400 ValidationException
- `stream_error`: streaming APIs send one delta, then a `modelStreamErrorException`;
  non-streaming APIs behave like `error`

For embeddings, error modes return the mapped error, `stream_error` returns 500
InternalServerException, `stall` sleeps then answers, and `truncate` / other modes return a normal
embedding.

**Scenarios** (`--scenarios file.json`) take precedence over modes. The first entry whose
`match` fits wins. `match.model` and `match.prompt` are regexes (`re.search`) against the model
ID and the full prompt (system + messages). Ids must be unique.

```json
{"scenarios": [
  {"id": "s1", "match": {"prompt": "refund"}, "reply": {"text": "No refunds."}},
  {"id": "s2", "match": {"model": "haiku"},
   "reply": {"tool_use": {"name": "lookup", "input": {"q": "x"}}}},
  {"id": "s3", "match": {"prompt": "boom"},
   "reply": {"status": 503, "error_type": "ServiceUnavailableException", "message": "down"}}
]}
```

Scenarios match embedding requests too (`match.model`, and `match.prompt` against the input text;
`times`, `replies` and `mode` work). A reply spec counts as an error only with `status` >= 400 or
`error_type`; `text`, `tool_use` and other statuses are ignored and a normal vector is returned.

`reply` keys: `text`, `tool_use` (`name`, `input`, optional `id`), `stop_reason` (default
`end_turn`, or `tool_use` with a tool call), `status`, `error_type`, `message`, `mode`.

Sequences: a scenario can also have
- `times` (positive int): matches at most that many times, then is skipped and matching continues
  with later scenarios, then modes.
- `replies` (non-empty list, instead of `reply`, not both): used in order on successive matches;
  the last one repeats.
- reply `mode`: one of the common modes above (`answer`, `throttle`, `stream_error`, ...). The reply
  comes from the normal mode machinery (`stall` uses `_stall_seconds`); other keys in that reply
  object are ignored.

```json
{"scenarios": [
  {"id": "flaky", "times": 2, "reply": {"mode": "throttle"}},
  {"id": "loop", "match": {"prompt": "weather"}, "times": 3, "replies": [
    {"tool_use": {"name": "get_city", "input": {}}},
    {"tool_use": {"name": "get_weather", "input": {"city": "Oslo"}}},
    {"text": "It is 5C in Oslo."}]}
]}
```

Per-scenario match counters are cleared by `reset()` (`POST /_fakerock/reset`) and whenever the
scenario list is replaced (`set_scenarios`). A scenario skipped because `times` ran out does not count.
Matched requests log `mode: "scenario"` plus `scenario: <id>`.

## Validation and bad config

Never a 500 (SDKs retry those). Returns 400 `ValidationException` for: empty body, non-object
JSON, Anthropic body without a `messages` list, Converse body without a `messages` list,
non-integer `max_tokens` / `inferenceConfig.maxTokens`, embedding body without `texts`
(Cohere) / `inputText` (Titan), bad Titan v2 `dimensions`, non-boolean Titan v2 `normalize`, Titan v2 `embeddingTypes` that is not a
non-empty list of `"float"` / `"binary"`. Reported output tokens are capped at
`max_tokens`. An invalid `modes.json` (bad JSON, family not an object, mode not a string) is
ignored: the last good modes stay in force and one warning is printed when the file changes.
An invalid scenarios file (duplicate `id`, bad regex, `tool_use` without a non-empty `name` or
with non-object `input`, non-integer `status`) stops startup with a message naming the scenario.

Error modes look real to AWS SDKs: throttle, 503 and 500 are retried by boto3 (standard and legacy
modes), validation is not (`tests/test_boto3_retries.py`). Throttle twice, then succeed:

```python
with MockBedrock(scenarios=[{"id": "f", "times": 2, "reply": {"mode": "throttle"}}]) as fr:
    fr.client().converse(modelId=m, messages=msgs); assert len(fr.requests()) == 3
```

Streaming responses use HTTP/1.1 chunked encoding (one chunk per event); others carry
`content-length`; connections are keep-alive.

## Logs

`<log-dir>/requests.jsonl`, one JSON object per request: `at` (epoch seconds), `api`
(`invoke`, `invoke-stream`, `converse`, `converse-stream`), `path`, `model`, `family`, `mode`
(`scenario` when a scenario matched), `status`, `prompt`. Embedding calls log family
`embedding` and `texts` (count) instead of `prompt`; they carry `mode` (label) and `scenario` like
chat rows. Malformed bodies log status 400 with null
family/mode. Every row also has `seq` (1-based, restarts on reset), `request_id` (equals the
response's `x-amzn-RequestId`), `body` (parsed request JSON; null if missing/unparseable, then
`raw` holds the first 2000 chars), `stream` (bool) and `scenario` (id, or null). Bodies are logged
in full, so prompts and images land in the log dir. Packs may write extra `*.jsonl` files to the same directory. The same rows are
available via `GET /_fakerock/requests` (in memory, last 10,000).

## Control API

For non-Python tests or Docker. JSON bodies; control calls are not logged.

- `GET /_fakerock/requests` -> `{"requests": [...]}`
- `PUT|POST /_fakerock/scenarios` (list or `{"scenarios": [...]}`) -> `{"ok": true, "count": n}`
- `PUT|POST /_fakerock/modes` (modes object) -> `{"ok": true}`
- `POST /_fakerock/reset` -> `{"ok": true}`

Invalid input -> 400 `ValidationException`; unknown path -> 404; wrong method -> 405.

## Packs and writing your own

- `generic` (default): if tools are offered and the last message is not a tool result, it
  calls the first tool, filling required properties with sample values (`"mock"`, `0`,
  `false`, `[]`, `{}`, or the first enum value). Otherwise it replies
  `Mock reply to: <last user text>` (first 200 chars).
- Your own: `--pack path/to/mypack.py`. This is how you add app-specific replies (match on
  the prompt, return structured output or tool calls). The file must define `PACK`, a `Pack` subclass with
  `family(call)`, `default_mode(family)`, `mode_key(call, family)` and
  `reply(call, family, mode, ctx)`. `ctx.log(name, row)` writes an extra jsonl file;
  `--pack-opt` values arrive in `self.options`.

```python
from mock_bedrock.packs.base import Pack, Reply

class Shout(Pack):
    name = "shout"
    def reply(self, call, family, mode, ctx):
        return Reply(text=call.last_user_text.upper())

PACK = Shout
```

In Docker, mount the file into the container (the working dir is `/app`, so `mock_bedrock`
imports).

## Embeddings caveat

Vectors are deterministic hashed bag-of-words. Default dims: `amazon.titan-embed-text-v1` 1536,
`amazon.titan-embed-text-v2*` 1024 (honours `dimensions` 256/512/1024), Cohere 1024.
Titan v2 also honours `normalize` (default true) and `embeddingTypes` (default `["float"]`): the
response has `embeddingsByType` with the requested types (binary = 1 if x > 0 else 0), and the
top-level `embedding` is omitted when only `binary` is requested.
`--embed-dims` / `MOCK_BEDROCK_EMBED_DIMS` overrides the default for all models. Similarity reflects word overlap only, not meaning. Re-index any stored
vectors after switching back to real Bedrock.

## Not covered

Answer quality, real latency/rate limits, token limits (counts are `len(text)//4`),
model-specific request validation, IAM/auth.

Client coverage: contract-tested with boto3 only. The Java SDK v2 and JavaScript SDK v3 should
work (same wire format and HTTP/1.1 chunked streams) but have not been tested yet.

## Tests

```
pip install -r requirements-dev.txt   # boto3, for the contract tests
python -m unittest discover -s tests
```

Without boto3 the contract tests are skipped and the rest still run. CI runs the suite on
Python 3.10 and 3.12 for every push and pull request.

## License

MIT. See [LICENSE](LICENSE).
