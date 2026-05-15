"""BM25 sparse vector encoder for hybrid retrieval.

Wraps fastembed's `Qdrant/bm25` model into the same `encode_passages` /
`encode_query` shape as `store.py`'s dense encoder. Uses Qdrant's IDF
modifier so per-token weights are computed server-side from the collection.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

BM25_MODEL_NAME = "Qdrant/bm25"


@lru_cache(maxsize=1)
def get_bm25_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=BM25_MODEL_NAME)


def _to_qdrant_sparse(emb) -> SparseVector:
    return SparseVector(
        indices=emb.indices.tolist(),
        values=emb.values.tolist(),
    )


def encode_passages_sparse(texts: list[str]) -> list[SparseVector]:
    model = get_bm25_model()
    return [_to_qdrant_sparse(e) for e in model.passage_embed(texts)]


def encode_query_sparse(text: str) -> SparseVector:
    model = get_bm25_model()
    embeddings = list(model.query_embed([text]))
    return _to_qdrant_sparse(embeddings[0])
