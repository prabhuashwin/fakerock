# Makes the `fakerock` fixture available when running from a checkout without
# `pip install`; once fakerock is installed its pytest plugin provides it anyway.
from mock_bedrock.pytest_plugin import fakerock  # noqa: F401
