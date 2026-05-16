"""Threshold calibration: sweep ``score_threshold`` against a labelled
query set, print a precision / recall / F1 table.

The user supplies a YAML file (``query → expected source``) — typically
20–30 queries pulled from real session transcripts. agent-kms runs each
query at every threshold in the sweep, scores retrieved chunks against
the gold set, and prints which threshold maximises F1.

Why this exists:
    The default ``score_threshold`` is tuned for a specific embedding
    model + corpus combination. Swapping the model or moving to a
    different corpus shifts the cosine distribution; the old threshold
    no longer separates relevant from noise. Without calibration, users
    either get empty results (threshold too high) or noise-bloated
    context (threshold too low).

Chunk identity:
    A retrieved chunk matches gold when ``basename(source) == gold.source``
    and (if gold specifies ``heading``) the heading matches. The
    basename-only path means gold entries don't have to track the
    absolute filesystem path the embedding pipeline records.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


def _chunk_key(item: dict, heading_specified: bool) -> tuple[str, str | None]:
    source = Path(item.get("source", "")).name
    if heading_specified:
        return (source, item.get("heading", ""))
    return (source, None)


def _gold_set(query: dict) -> set[tuple[str, str | None]]:
    """Build the gold tuple set for a query. If a gold entry omits
    ``heading``, match on source basename only (any heading counts)."""
    out: set[tuple[str, str | None]] = set()
    for g in query.get("gold", []):
        src = Path(g["source"]).name
        if "heading" in g:
            out.add((src, g["heading"]))
        else:
            out.add((src, None))
    return out


def _hits(retrieved: list[dict], gold: set[tuple[str, str | None]]) -> int:
    """Count retrieved chunks that satisfy any gold entry."""
    # Pre-split gold into heading-required vs source-only for one pass.
    src_only = {s for (s, h) in gold if h is None}
    src_heading = {(s, h) for (s, h) in gold if h is not None}
    n = 0
    for r in retrieved:
        src = Path(r.get("source", "")).name
        if src in src_only:
            n += 1
            continue
        if (src, r.get("heading", "")) in src_heading:
            n += 1
    return n


def precision_recall_f1(
    retrieved: list[dict],
    gold: set[tuple[str, str | None]],
) -> tuple[float, float, float]:
    if not gold:
        return (0.0, 0.0, 0.0)
    n_hits = _hits(retrieved, gold)
    p = n_hits / len(retrieved) if retrieved else 0.0
    r = n_hits / len(gold)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return (p, r, f1)


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Load eval queries from YAML.

    Schema (per item):
        id: str (optional, defaults to index)
        query: str (required)
        gold:
          - source: <basename or path; basename used for matching>
            heading: <H2 title; optional — omit to match any chunk from the file>
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(
            f"{path}: top-level must be a list of query objects"
        )
    for i, q in enumerate(data):
        if "query" not in q:
            raise ValueError(f"{path}: item {i} missing required field 'query'")
        if "gold" not in q or not q["gold"]:
            raise ValueError(
                f"{path}: item {i} ({q.get('id', '?')}) missing or empty 'gold'"
            )
        q.setdefault("id", str(i))
    return data


def sweep(
    queries: list[dict],
    thresholds: Iterable[float],
    retrieve_fn,
) -> list[dict]:
    """Run each query at each threshold; return one aggregate row per
    threshold (averaged P / R / F1, total returned chunks).
    """
    rows: list[dict] = []
    thresholds = list(thresholds)
    for t in thresholds:
        ps: list[float] = []
        rs: list[float] = []
        f1s: list[float] = []
        total_returned = 0
        empty_queries = 0
        for q in queries:
            retrieved = retrieve_fn(q["query"], score_threshold=t)
            total_returned += len(retrieved)
            if not retrieved:
                empty_queries += 1
            g = _gold_set(q)
            p, r, f1 = precision_recall_f1(retrieved, g)
            ps.append(p)
            rs.append(r)
            f1s.append(f1)
        n = len(queries)
        rows.append(
            {
                "threshold": t,
                "precision": sum(ps) / n,
                "recall": sum(rs) / n,
                "f1": sum(f1s) / n,
                "avg_returned": total_returned / n,
                "empty_queries": empty_queries,
            }
        )
    return rows


def format_table(rows: list[dict]) -> str:
    """Human-readable table for stdout."""
    if not rows:
        return "(no rows)"
    header = (
        f"{'T':>6}  {'P':>6}  {'R':>6}  {'F1':>6}  "
        f"{'avg_n':>7}  {'empty':>6}"
    )
    sep = "-" * len(header)
    best_f1 = max(r["f1"] for r in rows)
    lines = [header, sep]
    for r in rows:
        marker = " ◀ best F1" if r["f1"] == best_f1 and best_f1 > 0 else ""
        lines.append(
            f"{r['threshold']:>6.2f}  "
            f"{r['precision']:>6.3f}  "
            f"{r['recall']:>6.3f}  "
            f"{r['f1']:>6.3f}  "
            f"{r['avg_returned']:>7.1f}  "
            f"{r['empty_queries']:>6d}"
            f"{marker}"
        )
    return "\n".join(lines)


def frange(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range; rounds to 4dp to avoid 0.83000000001 noise."""
    out: list[float] = []
    n = 0
    while True:
        v = round(start + n * step, 4)
        if v > stop + 1e-9:
            break
        out.append(v)
        n += 1
    return out


def run(
    queries_path: Path,
    t_min: float,
    t_max: float,
    t_step: float,
    retrieve_fn=None,
) -> str:
    """End-to-end: load queries, sweep, return formatted table."""
    if retrieve_fn is None:
        from .retrieve import retrieve as _retrieve

        retrieve_fn = _retrieve
    queries = load_queries(queries_path)
    thresholds = frange(t_min, t_max, t_step)
    rows = sweep(queries, thresholds, retrieve_fn)
    return format_table(rows)
