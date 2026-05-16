"""Shared infra: embedding model, Qdrant client, collection schema, IDs.

All other modules (`ingest`, `server`, `session_extract`) consume these helpers
so prefix discipline (`passage: ` vs `query: `) and collection layout stay
in one place.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

# .env auto-load is now done in `agent_kms/__init__.py` so any module
# (llm, retrieve, server, session_extract) gets keys without depending on store.

COLLECTION = os.environ.get("AGENT_KMS_COLLECTION", "agent_knowledge")
MODEL_NAME = os.environ.get("AGENT_KMS_MODEL", "cl-nagoya/ruri-v3-310m")
VECTOR_SIZE = int(os.environ.get("AGENT_KMS_VECTOR_SIZE", "768"))


def _resolve_collection() -> str:
    """Resolve the active collection name.

    Precedence: ``AGENT_KMS_COLLECTION`` env > project ``kms.toml``
    ``[store].collection`` > ``"agent_knowledge"`` default. The module-level
    ``COLLECTION`` constant only honours env + default, so callers that need
    project-level config (``ensure_collection``, ``retrieve._collection``)
    must go through this helper. Resolved on every call so a single Python
    process can switch projects via ``chdir`` + ``config.reset_cache``.
    """
    env = os.environ.get("AGENT_KMS_COLLECTION", "")
    if env:
        return env
    try:
        # Local import to avoid a circular dependency at module load time.
        from .config import load_config

        return load_config().collection
    except Exception:
        return "agent_knowledge"

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    m = SentenceTransformer(MODEL_NAME)
    # Cap the sequence length to keep attention memory bounded. ModernBERT-
    # based models (e.g. ``cl-nagoya/ruri-v3-*``) ship with
    # ``max_seq_length=8192``; SDPA attention then allocates tens of GiB of
    # buffers and OOMs on CPU and consumer GPUs. The ingest pipeline tokenises
    # H2 sections that are typically a few hundred characters, so 512 is more
    # than enough. Models that already specify ≤512 (e.g. e5-base) are left
    # untouched.
    if getattr(m, "max_seq_length", 0) and m.max_seq_length > 512:
        m.max_seq_length = 512
    return m


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(reset: bool = False) -> None:
    name = _resolve_collection()
    client = get_client()
    exists = client.collection_exists(name)
    if exists and reset:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def stable_id(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def encode_passage(text: str) -> list[float]:
    return get_model().encode("passage: " + text, normalize_embeddings=True).tolist()


def encode_passages(texts: list[str]) -> list[list[float]]:
    arr = get_model().encode(
        ["passage: " + t for t in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return arr.tolist()


def encode_query(text: str) -> list[float]:
    return get_model().encode("query: " + text, normalize_embeddings=True).tolist()
