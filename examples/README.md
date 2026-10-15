# fakerock examples

- `app.py` - tiny app code: a boto3 `bedrock-runtime` client with a retry policy (3 attempts), `ask` and `ask_stream`.
- `test_retry_recipe.py` - pytest recipe using the `fakerock` fixture: throttle-then-succeed, give up after max attempts, no retry on validation errors, mid-stream errors. Backoff sleeps are patched out so it runs instantly.

Run:

    pip install "fakerock[test] @ git+https://github.com/prabhuashwin/fakerock"   # or: pip install -e ".[test]"
    python -m pytest examples -q

For CI use, see github-actions.yml.
