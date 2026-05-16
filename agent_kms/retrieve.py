"""Retrieval pipeline (threshold-based, no count cap).

Returns every chunk whose **boosted cosine score crosses `score_threshold`**.
There is no top-N cap — relevance alone decides inclusion.

Components:
  1. Template query expansion — append fixed universal-aspects to query
     so embeddings catch asset / completion / faithful-port docs.
  2. Severity boost — additive score adjustment per chunk payload tag:
       critical: +0.05, high: +0.025, default: 0
     Acts as a relevance signal weighted by curated importance.
  3. Threshold filter — boosted score must be >= score_threshold.
  4. (Implicit) Qdrant ID-level dedup — one chunk per stored point.

`retrieve_simple()` keeps the unenhanced cosine path for tests.

Note (2026-05-11): switched from count-based (`limit`) to threshold-based
(`score_threshold`). The previous `universal_floor` mechanism was removed
because it inflated count for off-axis queries — that contradicts the
"relevance only" principle. critical/universal docs still surface via
severity_boost when they're topically close.
"""

from __future__ import annotations

from .query_expand import template_expand
from .store import encode_query, get_client


def _collection() -> str:
    # Resolved lazily so AGENT_KMS_COLLECTION env AND project kms.toml
    # ``[store].collection`` are both honoured (env wins).
    from .store import _resolve_collection
    return _resolve_collection()

# severity → additive score boost. Tuned for cosine ranges 0.78-0.92.
DEFAULT_SEVERITY_BOOST = {
    "critical": 0.05,
    "high": 0.025,
    "default": 0.0,
}

# applicability → additive score boost. Penalises narrow-topic chunks so they
# only surface when cosine is strongly aligned (precision uplift without
# sacrificing recall on truly relevant matches). universal chunks (= apply to
# ANY context) keep their cosine intact; conditional chunks lose a hair;
# topic-specific chunks need a clear topical hit instead of riding the corpus
# baseline.
DEFAULT_APPLICABILITY_BOOST = {
    "universal": 0.0,
    "conditional": -0.005,
    "topic-specific": -0.015,
}

# Default cosine + boost score floor. Calibrate per corpus with
# ``agent-kms calibrate-threshold`` against a small labelled query set.
#
# 0.83 is tuned for ``cl-nagoya/ruri-v3-310m`` (the current default
# embedding model) on Japanese technical-documentation corpora — empirical
# F1-optimal in the 0.81–0.85 band on the spin-glass/rag-evaluation-jp
# benchmark. If your corpus is predominantly English, or you swap to a
# different model via ``AGENT_KMS_MODEL`` / ``score_threshold`` in
# ``kms.toml``, recalibrate. The legacy default for
# ``intfloat/multilingual-e5-base`` was 0.93 (higher Japanese cosine
# baseline) — set that explicitly if reverting the model.
DEFAULT_SCORE_THRESHOLD = 0.83

# Safety cap on initial fetch. Threshold filters most, but prevents
# pathological queries from pulling thousands of chunks.
DEFAULT_MAX_FETCH = 200


def _payload_to_dict(p) -> dict:
    return {
        "text": p.payload.get("text", ""),
        "source": p.payload.get("source_file", ""),
        "source_type": p.payload.get("source_type", ""),
        "severity": p.payload.get("severity", "default"),
        "applicability": p.payload.get("applicability", "topic-specific"),
        "heading": p.payload.get("heading", ""),
        "score": float(p.score),
    }


def retrieve_simple(query: str, limit: int = 5) -> list[dict]:
    """Vanilla cosine top-N (no expand / boost / threshold). For tests."""
    vec = encode_query(query)
    points = (
        get_client()
        .query_points(collection_name=_collection(), query=vec, limit=limit)
        .points
    )
    return [_payload_to_dict(p) for p in points]


def retrieve(
    query: str,
    *,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    expand: bool = True,
    severity_boost: dict | None = None,
    applicability_boost: dict | None = None,
    max_fetch: int = DEFAULT_MAX_FETCH,
) -> list[dict]:
    """Relevance-threshold retrieve. No count cap.

    Returns every chunk whose (cosine + severity_boost) >= score_threshold,
    sorted by score descending.

    Args:
        score_threshold: boosted score floor. Chunks below are dropped.
        expand: apply template_expand on query (lifts universal-aspect docs).
        severity_boost: per-severity additive boost. Pass `{}` to disable
            for pure cosine. Default: critical +0.05, high +0.025.
        max_fetch: initial pull size from Qdrant (safety cap against very
            low thresholds returning thousands of items).

    Returns:
        List of dicts, descending score order. May be empty if no chunk
        crosses the threshold.

    Backward compat:
        Callers passing `limit=N` should migrate to `score_threshold=...`.
        A positional `limit` argument is still accepted for migration grace
        and is interpreted as a hint for `max_fetch` only — the count is
        NOT capped to `limit`.
    """
    boost = severity_boost if severity_boost is not None else DEFAULT_SEVERITY_BOOST
    app_boost = (
        applicability_boost if applicability_boost is not None
        else DEFAULT_APPLICABILITY_BOOST
    )
    eff_query = template_expand(query) if expand else query
    vec = encode_query(eff_query)
    client = get_client()

    # Over-fetch up to max_fetch; threshold filtering does the real work.
    main = client.query_points(
        collection_name=_collection(),
        query=vec,
        limit=max_fetch,
    ).points

    # Apply additive boosts: severity (positive) + applicability (negative for
    # narrow-topic). Both signed adjustments combined.
    for p in main:
        sev = p.payload.get("severity", "default")
        app = p.payload.get("applicability", "topic-specific")
        delta = boost.get(sev, 0.0) + app_boost.get(app, 0.0)
        if delta != 0.0:
            p.score = float(p.score) + delta

    # Filter by threshold, then sort descending by boosted score
    passed = [p for p in main if float(p.score) >= score_threshold]
    passed.sort(key=lambda p: -float(p.score))

    return [_payload_to_dict(p) for p in passed]
