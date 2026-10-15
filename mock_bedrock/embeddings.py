"""Deterministic word-hash embeddings (texts sharing words are close)."""
import hashlib
import math
import re
import uuid

STOP = set("what who how the and for are was does can could would with from about you your this that have has when where which will there".split())


def embed(text: str, dims: int = 1024, normalize: bool = True) -> list:
    words = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP]
    vec = [0.0] * dims
    if not words:
        vec = [0.01] * dims
    for w in words:
        h = hashlib.sha256(w.encode("utf-8")).digest()
        vec[int.from_bytes(h[:4], "big") % dims] += 1.0 if h[4] & 1 else -1.0
    if not normalize:
        return vec
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def is_embedding_model(model: str) -> bool:
    m = model.split("/")[-1]
    return m.startswith("cohere.embed") or m.startswith("amazon.titan-embed")


def default_dims(model: str) -> int:
    m = model.split("/")[-1]
    return 1536 if m.startswith("amazon.titan-embed-text-v1") else 1024


def handle_embedding(model: str, req: dict, dims_override=None):
    """Return (response dict, item count). Raises ValueError on a malformed request.

    dims_override (--embed-dims) replaces the per-model default; Titan v2 still honours
    a request `dimensions` field (256/512/1024), `normalize` (bool, default True) and
    `embeddingTypes` (non-empty list of "float"/"binary", default ["float"]); the response
    then carries `embeddingsByType`, plus top-level `embedding` only when "float" is requested.
    """
    m = model.split("/")[-1]
    dims = dims_override or default_dims(m)
    if m.startswith("cohere.embed"):
        texts = req.get("texts")
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("texts must be a list of strings")
        return ({"id": str(uuid.uuid4()), "texts": texts,
                 "embeddings": [embed(t, dims) for t in texts],
                 "response_type": "embeddings_floats"}, len(texts))
    text = req.get("inputText")
    if not isinstance(text, str):
        raise ValueError("inputText must be a string")
    if m.startswith("amazon.titan-embed-text-v2"):
        if req.get("dimensions") is not None:
            d = req["dimensions"]
            if isinstance(d, bool) or d not in (256, 512, 1024):
                raise ValueError("dimensions must be one of 256, 512, 1024")
            dims = d
        normalize = req.get("normalize", True)
        if not isinstance(normalize, bool):
            raise ValueError("normalize must be a boolean")
        types = req.get("embeddingTypes", ["float"])
        if (not isinstance(types, list) or not types
                or not all(isinstance(t, str) and t in ("float", "binary") for t in types)):
            raise ValueError("embeddingTypes must be a non-empty list of 'float' and/or 'binary'")
        vec = embed(text, dims, normalize)
        by_type = {}
        if "float" in types:
            by_type["float"] = vec
        if "binary" in types:
            by_type["binary"] = [1 if x > 0 else 0 for x in vec]
        resp = {"inputTextTokenCount": max(1, len(text.split())), "embeddingsByType": by_type}
        if "float" in types:
            resp["embedding"] = vec
        return resp, 1
    return ({"embedding": embed(text, dims),
             "inputTextTokenCount": max(1, len(text.split()))}, 1)
