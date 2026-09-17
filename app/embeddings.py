"""Local text embeddings via fastembed (all-MiniLM-L6-v2, 384 dims, L2-normed).

Runs fully offline on CPU (ONNX) — no per-call API cost, matching the Phase 1
decision in the build spec. The model is loaded lazily as a process-wide
singleton and shared across requests.
"""

import logging
import os
import threading

from fastembed import TextEmbedding

from .settings import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Must be set before tokenizers spawns threads; avoids fork warnings in servers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_LOCK = threading.Lock()
_MODEL: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                logger.info("loading embedding model %s", EMBEDDING_MODEL)
                _MODEL = TextEmbedding(EMBEDDING_MODEL)
    return _MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts into 384-dim L2-normalized vectors, preserving order."""
    if not texts:
        return []
    return [vec.tolist() for vec in _get_model().embed(texts)]


# A MiniLM embedding blurs badly past a few hundred words: one vector cannot
# represent a whole article, so its middle becomes unretrievable. Long text is
# split on sentence boundaries into overlapping windows instead.
_CHUNK_CHARS = 700
_CHUNK_OVERLAP_CHARS = 120


def split_for_embedding(text: str) -> list[str]:
    """Split long text into overlapping chunks; short text passes through.

    Splits at sentence ends where possible so a chunk is never cut mid-fact,
    and overlaps consecutive chunks so a fact spanning a boundary still lives
    somewhere complete.
    """
    cleaned = " ".join(text.split())
    if len(cleaned) <= _CHUNK_CHARS:
        return [cleaned] if cleaned else []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + _CHUNK_CHARS, len(cleaned))
        if end < len(cleaned):
            boundary = max(
                cleaned.rfind(". ", start, end),
                cleaned.rfind("! ", start, end),
                cleaned.rfind("? ", start, end),
            )
            if boundary > start + _CHUNK_CHARS // 2:
                end = boundary + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(end - _CHUNK_OVERLAP_CHARS, start + 1)
    return chunks
