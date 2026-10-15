"""Tiny 'application code' that talks to Bedrock. The recipe tests exercise this module."""
import boto3
from botocore.config import Config

MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


def make_client(endpoint_url=None, **kw):
    """Build the bedrock-runtime client with the app's retry policy.

    In real use leave endpoint_url=None; boto3 then honours the env var
    AWS_ENDPOINT_URL_BEDROCK_RUNTIME (point it at fakerock locally).
    """
    cfg = Config(retries={"mode": "standard", "total_max_attempts": 3})
    return boto3.client("bedrock-runtime", region_name="us-east-1",
                        endpoint_url=endpoint_url, config=cfg, **kw)


def _messages(prompt):
    return [{"role": "user", "content": [{"text": prompt}]}]


def ask(client, prompt, model=MODEL):
    """One-shot call; returns the reply text."""
    r = client.converse(modelId=model, messages=_messages(prompt))
    return r["output"]["message"]["content"][0]["text"]


def ask_stream(client, prompt, model=MODEL):
    """Streaming call; joins the text deltas into one string."""
    r = client.converse_stream(modelId=model, messages=_messages(prompt))
    parts = []
    for event in r["stream"]:
        if "contentBlockDelta" in event:
            parts.append(event["contentBlockDelta"]["delta"].get("text", ""))
    return "".join(parts)
