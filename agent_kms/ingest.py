"""Config-driven ingest: read kms.toml ``[[ingest.sources]]`` and upsert.

Replaces source-specific ingest entry-points. Idempotent via deterministic
SHA256 IDs; re-running upserts the same points in place. ``--reset`` drops
the collection first.

Source-type tagging:
  - ``source_type`` defaults to the source ``name`` from kms.toml so callers
    can keep filtering by named buckets (e.g. ``"known_issue"``, ``"adr"``).
  - ``severity`` / ``applicability`` defaults come from the source entry,
    overridden per-file via ``[file_severity]`` table in kms.toml (path key
    is relative to ``knowledge_root``).
  - For YAML sources, the ``frontmatter_field`` option enables reading a
    YAML field (default ``domain``) and mapping it to ``applicability`` via
    ``[domain_applicability]`` in kms.toml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from qdrant_client.models import PointStruct

from .chunker import CHUNKERS, Chunk
from .config import KMSConfig, load_config
from .store import encode_passages, ensure_collection, get_client, stable_id


def _resolve_root(cfg: KMSConfig, rel: str) -> Path:
    p = Path(rel).expanduser()
    return p if p.is_absolute() else (cfg.knowledge_root / p)


def _apply_file_overrides(cfg: KMSConfig, chunk: Chunk) -> Chunk:
    try:
        rel = str(Path(chunk.source_file).resolve().relative_to(cfg.knowledge_root))
    except ValueError:
        rel = chunk.source_file
    override = cfg.file_severity_map.get(rel)
    if override:
        chunk.severity, chunk.applicability = override
    return chunk


def _parse_yaml_frontmatter(text: str) -> dict:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        body = text[3:end] if end > 0 else ""
    else:
        body = text
    try:
        data = yaml.safe_load(body)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _apply_yaml_metadata(cfg: KMSConfig, chunk: Chunk, source_opts: dict) -> Chunk:
    field = source_opts.get("frontmatter_field", "domain")
    meta = _parse_yaml_frontmatter(chunk.body)
    val = meta.get(field, "")
    if val and val in cfg.domain_applicability_map:
        chunk.applicability = cfg.domain_applicability_map[val]
    sev_field = source_opts.get("severity_field")
    if sev_field and meta.get(sev_field):
        chunk.severity = str(meta[sev_field])
    conf_field = source_opts.get("confidence_field")
    if conf_field and meta.get(conf_field) is not None:
        try:
            chunk.confidence = float(meta[conf_field])
        except (TypeError, ValueError):
            pass
    if "tags" in meta and isinstance(meta["tags"], list):
        chunk.tags = [str(t) for t in meta["tags"]]
    return chunk


def collect_chunks(cfg: KMSConfig) -> list[Chunk]:
    out: list[Chunk] = []
    for src in cfg.sources:
        fn = CHUNKERS.get(src.kind)
        if fn is None:
            print(f"  warning: unknown chunker kind {src.kind!r} for source {src.name!r}")
            continue
        for rel in src.roots:
            root = _resolve_root(cfg, rel)
            glob = src.options.get("glob") or (
                "*.md" if src.kind == "markdown_h2" else "*.yaml"
            )
            if src.kind == "markdown_h2":
                iterator = fn(
                    root,
                    glob=glob,
                    min_chars=src.min_chars,
                    default_severity=src.default_severity,
                    default_applicability=src.default_applicability,
                    default_confidence=src.default_confidence,
                    source_type=src.name,
                )
            else:
                iterator = fn(
                    root,
                    glob=glob,
                    default_severity=src.default_severity,
                    default_applicability=src.default_applicability,
                    default_confidence=src.default_confidence,
                    source_type=src.name,
                )
            for chunk in iterator:
                chunk = _apply_file_overrides(cfg, chunk)
                if src.kind == "yaml_file":
                    chunk = _apply_yaml_metadata(cfg, chunk, src.options)
                out.append(chunk)
    return out


def upsert_chunks(cfg: KMSConfig, chunks: list[Chunk], *, batch: int = 64) -> int:
    if not chunks:
        return 0
    ensure_collection()
    client = get_client()
    total = 0
    for i in range(0, len(chunks), batch):
        slice_ = chunks[i : i + batch]
        vectors = encode_passages([c.embed_text for c in slice_])
        points = []
        for c, vec in zip(slice_, vectors):
            payload = {
                "text": c.body,
                "heading": c.heading,
                "source_file": c.source_file,
                "source_type": c.source_type,
                "severity": c.severity,
                "applicability": c.applicability,
                "confidence": c.confidence,
                "tags": c.tags,
            }
            # Optional frontmatter-derived fields — only added when present
            # so legacy chunks without a frontmatter block keep their
            # original payload shape and stable_id rebuild stays minimal.
            if c.source_pr:
                payload["source_pr"] = c.source_pr
            if c.captured_at:
                payload["captured_at"] = c.captured_at
            points.append(
                PointStruct(
                    id=stable_id(f"{c.source_type}:{c.source_file}#{c.heading}"),
                    vector=vec,
                    payload=payload,
                )
            )
        client.upsert(collection_name=cfg.collection, points=points)
        total += len(points)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description="agent-kms ingest")
    ap.add_argument("--preset", default=None, help="preset name (default: general)")
    ap.add_argument("--reset", action="store_true", help="drop collection first")
    args = ap.parse_args()

    cfg = load_config(args.preset)
    print("agent-kms ingest")
    print(f"  preset:         {cfg.preset}")
    print(f"  knowledge_root: {cfg.knowledge_root}")
    print(f"  collection:     {cfg.collection}")
    print(f"  sources:        {[s.name for s in cfg.sources]}")

    if args.reset:
        ensure_collection(reset=True)
        print("  reset:          collection dropped")
    else:
        ensure_collection()

    chunks = collect_chunks(cfg)
    print(f"  collected:      {len(chunks)} chunks")
    n = upsert_chunks(cfg, chunks)
    print(f"  upserted:       {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
