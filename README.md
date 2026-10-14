# fakerock: a local mock server for AWS Bedrock Runtime

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
| `mock_bedrock/` | The server: routing and modes (`server.py`), InvokeModel (`invoke.py`), Converse (`converse.py`), AWS event-stream encoding (`eventstream.py`), embeddings (`embeddings.py`), live controls and request log (`control.py`) |
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
per-model default].

Docker:

```
docker build -t mock-bedrock .
docker run --rm -p 127.0.0.1:52999:52999 -v "$PWD/.mock-bedrock:/data" mock-bedrock
docker run ... mock-bedrock --scenarios /data/scenarios.json
```

`docker-compose.example.yml` shows a loopback-only publish and a sibling container. The image binds `0.0.0.0` inside
the container only; do not publish the port on a non-loopback host interface. In Docker change
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

## Controlling replies

The pack classifies each request into a **family** (generic: `chat`, or `tools` when the
request offers tools). The mode comes from `modes.json` in the log dir (or `--modes`),
**re-read on every call**, so you can edit it while the server runs:

```json
{
  "_stall_seconds": 60,
  "chat":  {"_default": "answer", "summarise": "throttle"},
  "tools": {"_default": "answer"}
}
```

Within a family, each key other than `_default` is a case-insensitive substring tested against
the pack's mode key (generic: last user message text); first match wins, else `_default`, else
the pack's default (`answer`). Packs may add their own families and modes.

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

`reply` keys: `text`, `tool_use` (`name`, `input`, optional `id`), `stop_reason` (default
`end_turn`, or `tool_use` with a tool call), `status`, `error_type`, `message`.

## Validation and bad config

Never a 500 (SDKs retry those). Returns 400 `ValidationException` for: empty body, non-object
JSON, Anthropic body without a `messages` list, Converse body without a `messages` list,
non-integer `max_tokens` / `inferenceConfig.maxTokens`, embedding body without `texts`
(Cohere) / `inputText` (Titan), bad Titan v2 `dimensions`. Reported output tokens are capped at
`max_tokens`. An invalid `modes.json` (bad JSON, family not an object, mode not a string) is
ignored: the last good modes stay in force and one warning is printed when the file changes.
An invalid scenarios file (duplicate `id`, bad regex, `tool_use` without a non-empty `name` or
with non-object `input`, non-integer `status`) stops startup with a message naming the scenario.

Streaming responses use HTTP/1.1 chunked encoding (one chunk per event); others carry
`content-length`; connections are keep-alive.

## Logs

`<log-dir>/requests.jsonl`, one JSON object per request: `at` (epoch seconds), `api`
(`invoke`, `invoke-stream`, `converse`, `converse-stream`), `path`, `model`, `family`, `mode`
(`scenario` when a scenario matched), `status`, `prompt`. Embedding calls log family
`embedding` and `texts` (count) instead of `prompt`. Malformed bodies log status 400 with null
family/mode. Packs may write extra `*.jsonl` files to the same directory.

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
`--embed-dims` / `MOCK_BEDROCK_EMBED_DIMS` overrides the default for all models. Similarity reflects word overlap only, not meaning. Re-index any stored
vectors after switching back to real Bedrock.

## Not covered

Answer quality, real latency/rate limits, token limits (counts are `len(text)//4`),
model-specific request validation, IAM/auth.

## Tests

```
python -m unittest discover -s tests
```

## License

MIT. See [LICENSE](LICENSE).
