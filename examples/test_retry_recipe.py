"""Recipe: test your app's retry behaviour against fakerock using the `fakerock` pytest fixture."""
from unittest import mock

import pytest
from botocore.exceptions import ClientError, EventStreamError

import app


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    # Skip botocore's real exponential backoff so retry tests run instantly.
    # (Patch both the "standard" and "legacy" retry implementations.)
    for target in ("botocore.retries.standard.ExponentialBackoff.delay_amount",
                   "botocore.retryhandler.delay_exponential"):
        monkeypatch.setattr(target, mock.Mock(return_value=0))


@pytest.fixture
def client(fakerock):
    # Build the client with the app's own factory so its retry config is what's tested.
    return app.make_client(endpoint_url=fakerock.endpoint_url,
                           aws_access_key_id="dummy", aws_secret_access_key="dummy")


def fail(mode, times=None):
    s = {"id": "f", "reply": {"mode": mode}}
    if times:
        s["times"] = times  # fail this many requests, then behave normally
    return [s]


def test_throttled_twice_then_succeeds(fakerock, client):
    fakerock.set_scenarios(fail("throttle", 2))
    assert app.ask(client, "hi") == "Mock reply to: hi"
    assert len(fakerock.requests()) == 3  # 2 throttled + 1 success


def test_throttled_forever_gives_up(fakerock, client):
    fakerock.set_scenarios(fail("throttle"))
    with pytest.raises(ClientError) as ei:
        app.ask(client, "hi")
    assert ei.value.response["Error"]["Code"] == "ThrottlingException"
    assert len(fakerock.requests()) == 3  # total_max_attempts


def test_validation_error_is_not_retried(fakerock, client):
    fakerock.set_scenarios(fail("validation"))
    with pytest.raises(ClientError) as ei:
        app.ask(client, "hi")
    assert ei.value.response["Error"]["Code"] == "ValidationException"
    assert len(fakerock.requests()) == 1


def test_stream_error_mid_stream(fakerock, client):
    # The stream starts fine, then fails; botocore raises EventStreamError while iterating.
    fakerock.set_scenarios(fail("stream_error"))
    with pytest.raises(EventStreamError) as ei:
        app.ask_stream(client, "hi")
    assert ei.value.response["Error"]["Code"] == "modelStreamErrorException"
