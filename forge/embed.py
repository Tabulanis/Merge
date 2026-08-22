"""Her associative sense: text → vectors, in her own space.

A small embedding model (nomic-embed-text) runs beside her brains on :8086 —
the organ that lets memory be found by MEANING, not just matching words. "the
rocket wall" finds "hit a ceiling on the magnetoplasma design" even though they
share no keywords. Only the embeddings are hers; what she's handed back is still
text (llama.cpp reads tokens, not raw vectors) — but it's *found* in her language.

Graceful by design: if the organ is down, every function returns None and callers
fall back to keyword search. Memory gets less associative, nothing breaks — the
same posture recall.py already takes when the librarian is asleep.

nomic wants task prefixes: documents are stored as "search_document: …", the
query is asked as "search_query: …". That asymmetry is what makes retrieval work.
"""
from __future__ import annotations

import httpx
import numpy as np

from .config import load_config

DIM = 768                      # nomic-embed-text-v1.5 output width
_DEFAULT_URL = "http://127.0.0.1:8086/v1"
_TIMEOUT = 20.0


def _base() -> str:
    m = (load_config().get("models") or {}).get("embed") or {}
    return (m.get("base_url") or _DEFAULT_URL).rstrip("/")


def available() -> bool:
    """Is the embedding organ up? Cheap health ping; False on any trouble."""
    try:
        root = _base()[:-3] if _base().endswith("/v1") else _base()
        return httpx.get(root + "/health", timeout=2.0).status_code == 200
    except Exception:
        return False


# The server refuses (or times out on) large batches: measured live, 16 texts
# took 12s and 32 came back a failure. Callers were passing far more than that —
# Cortex indexes in chunks of 64 — and got a silent None, so a whole archive
# could fail to embed with nothing said. Split the work here so every caller is
# safe rather than each having to know the limit.
_BATCH = 12


def _post(texts: list[str]) -> np.ndarray | None:
    """Raw call → L2-normalized rows (so a dot product IS cosine). None if down."""
    if not texts:
        return np.empty((0, DIM), dtype="float32")
    if len(texts) > _BATCH:
        out = []
        for i in range(0, len(texts), _BATCH):
            part = _post(texts[i:i + _BATCH])
            if part is None:
                return None          # a failed chunk fails the whole call, loudly
            out.append(part)
        return np.vstack(out)
    try:
        r = httpx.post(_base() + "/embeddings",
                       json={"input": texts, "model": "embed"}, timeout=_TIMEOUT)
        r.raise_for_status()
        rows = [d["embedding"] for d in r.json()["data"]]
        arr = np.asarray(rows, dtype="float32")
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
        return arr
    except Exception:
        return None


def embed_documents(texts: list[str]) -> np.ndarray | None:
    """Many memories → an (N, DIM) matrix of unit vectors, or None if the organ's down."""
    return _post([f"search_document: {t}" for t in texts])


def embed_query(text: str) -> np.ndarray | None:
    """One question → a single unit vector (DIM,), or None if the organ's down."""
    v = _post([f"search_query: {text}"])
    return None if v is None or len(v) == 0 else v[0]
