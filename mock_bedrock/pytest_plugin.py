"""pytest fixture `fakerock`. Enable with `pytest_plugins = ["mock_bedrock.pytest_plugin"]` in conftest.py.

Each test gets its own server, so it is safe with pytest-xdist.
"""
import pytest

from mock_bedrock.testing import MockBedrock


@pytest.fixture
def fakerock():
    with MockBedrock() as mock:
        yield mock
