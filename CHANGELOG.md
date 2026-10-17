# Changelog

All notable changes to fakerock. Versions are git tags `v<version>`; each tag publishes the
Docker image, the PyPI package and a GitHub release.

## Unreleased

- Stricter type checks on request bodies (non-string text blocks, non-list `content`/`system`/`tools`,
  non-object `toolConfig`) return `ValidationException` instead of a 500; scenario `reply` string fields
  and `tool_use.id` are validated at load; tool-use ids come from the shared id source.
- README: naming note, contributing section, changelog; CI tests Python 3.13; tags publish to PyPI and
  create a GitHub release.

## 0.4.0

- Mid-stream error modes and `_stream_error_after`; opt-in strict request validation
  (`--strict`); JS SDK v3 smoke test in CI; sdist now includes the test helpers.

## 0.3.0

- The full set of Bedrock error modes, `_latency_seconds`, `_chunk_delay_seconds`, and `drop`
  (closed connection) injection.

## 0.2.0

- First packaged release: `fakerock` command, `MockBedrock` test double, pytest fixture,
  control API, `--deterministic`, GHCR image.
