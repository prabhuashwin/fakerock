# fakerock examples

- `app.py` - tiny app code: a boto3 `bedrock-runtime` client with a retry policy (3 attempts), `ask` and `ask_stream`.
- `test_retry_recipe.py` - pytest recipe using the `fakerock` fixture: throttle-then-succeed, give up after max attempts, no retry on validation errors, mid-stream errors. Backoff sleeps are patched out so it runs instantly.

Run:

    pip install "fakerock[test] @ git+https://github.com/prabhuashwin/fakerock"   # or: pip install -e ".[test]"
    python -m pytest examples -q

For CI use, see github-actions.yml.

JavaScript (AWS SDK v3) smoke test, `js/smoke.mjs`. From the repo root, start two servers, then run it:

    python -m mock_bedrock --port 52999 &
    python -m mock_bedrock --port 52998 --strict &
    cd examples/js && npm ci
    FAKEROCK_URL=http://127.0.0.1:52999 FAKEROCK_STRICT_URL=http://127.0.0.1:52998 node smoke.mjs

The client passes `requestHandler: new NodeHttpHandler()` because the SDK defaults to HTTP/2 and fakerock is HTTP/1.1.
