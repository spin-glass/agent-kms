"""Project + preset configuration loader.

Resolves the following sources, in priority order (later wins):
  1. Built-in preset (``agent_kms/presets/<preset>/kms.toml``)
  2. Project config (``$AGENT_KMS_CONFIG`` or ``./.agent-kms/kms.toml``)
  3. Environment variables (``AGENT_KMS_COLLECTION`` etc — handled by store.py)

Also resolves prompt template paths so callers can swap session-extract /
query-expand prompts per project without editing Python code.

Knowledge root
==============
``AGENT_KMS_KNOWLEDGE_ROOT`` (env) → ``[knowledge].root`` (toml) → ``Path.cwd()``
The directories listed under ``[[ingest.sources]] roots`` are resolved
relative to this root.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
PRESETS_DIR = PACKAGE_DIR / "presets"
DEFAULT_PRESET = "general"


@dataclass
class IngestSource:
    name: str
    kind: str  # "markdown_h2" | "yaml_file" | "directory"
    roots: list[str]
    default_severity: str = "default"
    default_applicability: str = "topic-specific"
    default_confidence: float = 1.0
    min_chars: int = 100
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class KMSConfig:
    preset: str
    knowledge_root: Path
    collection: str
    sources: list[IngestSource]
    severity_boost: dict[str, float]
    applicability_boost: dict[str, float]
    score_threshold: float
    prompts_dir: Path
    forbidden_vocab: list[str]
    file_severity_map: dict[str, tuple[str, str]]
    domain_applicability_map: dict[str, str]
    raw: dict[str, Any]


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_preset(name: str) -> Path:
    p = PRESETS_DIR / name / "kms.toml"
    if not p.exists():
        raise FileNotFoundError(
            f"unknown preset {name!r}; expected {p}. "
            f"Available: {[d.name for d in PRESETS_DIR.iterdir() if d.is_dir()]}"
        )
    return p


def _project_config_path() -> Path | None:
    explicit = os.environ.get("AGENT_KMS_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    candidate = Path.cwd() / ".agent-kms" / "kms.toml"
    return candidate if candidate.exists() else None


@lru_cache(maxsize=1)
def load_config(preset: str | None = None) -> KMSConfig:
    preset_name = preset or os.environ.get("AGENT_KMS_PRESET", DEFAULT_PRESET)
    preset_path = _resolve_preset(preset_name)
    preset_data = _load_toml(preset_path)

    project_path = _project_config_path()
    project_data = _load_toml(project_path) if project_path else {}

    merged = _deep_merge(preset_data, project_data)

    knowledge_root = Path(
        os.environ.get(
            "AGENT_KMS_KNOWLEDGE_ROOT",
            merged.get("knowledge", {}).get("root", str(Path.cwd())),
        )
    ).expanduser().resolve()

    sources_raw = merged.get("ingest", {}).get("sources", [])
    sources = [
        IngestSource(
            name=s["name"],
            kind=s["kind"],
            roots=s.get("roots", []),
            default_severity=s.get("default_severity", "default"),
            default_applicability=s.get("default_applicability", "topic-specific"),
            default_confidence=float(s.get("default_confidence", 1.0)),
            min_chars=int(s.get("min_chars", 100)),
            options=s.get("options", {}),
        )
        for s in sources_raw
    ]

    retrieve = merged.get("retrieve", {})
    severity_boost = retrieve.get(
        "severity_boost", {"critical": 0.05, "high": 0.025, "default": 0.0}
    )
    applicability_boost = retrieve.get(
        "applicability_boost",
        {"universal": 0.0, "conditional": -0.005, "topic-specific": -0.015},
    )
    score_threshold = float(retrieve.get("score_threshold", 0.93))

    prompts_dir_str = merged.get("prompts", {}).get("dir")
    if prompts_dir_str:
        prompts_dir = Path(prompts_dir_str).expanduser()
        if not prompts_dir.is_absolute():
            prompts_dir = (project_path.parent if project_path else Path.cwd()) / prompts_dir
    else:
        prompts_dir = PRESETS_DIR / preset_name / "prompts"

    se = merged.get("session_extract", {})
    forbidden_vocab = se.get(
        "forbidden_vocab",
        ["skeleton", "placeholder", "minimal", "defer", "scope-narrow", "dummy"],
    )

    file_severity_map_raw = merged.get("file_severity", {})
    file_severity_map = {
        k: (v.get("severity", "default"), v.get("applicability", "topic-specific"))
        for k, v in file_severity_map_raw.items()
    }

    domain_applicability_map = merged.get("domain_applicability", {})

    collection = os.environ.get(
        "AGENT_KMS_COLLECTION",
        merged.get("store", {}).get("collection", "agent_knowledge"),
    )

    return KMSConfig(
        preset=preset_name,
        knowledge_root=knowledge_root,
        collection=collection,
        sources=sources,
        severity_boost=severity_boost,
        applicability_boost=applicability_boost,
        score_threshold=score_threshold,
        prompts_dir=prompts_dir,
        forbidden_vocab=forbidden_vocab,
        file_severity_map=file_severity_map,
        domain_applicability_map=domain_applicability_map,
        raw=merged,
    )


def load_prompt(name: str, preset: str | None = None) -> str:
    """Load a prompt template by name. Falls back to general preset."""
    cfg = load_config(preset)
    candidate = cfg.prompts_dir / f"{name}.md"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    fallback = PRESETS_DIR / DEFAULT_PRESET / "prompts" / f"{name}.md"
    if fallback.exists():
        return fallback.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"prompt {name!r} not found in {cfg.prompts_dir} or {fallback.parent}"
    )


def reset_cache() -> None:
    """Force reload on next access (for tests + agent-kms CLI subcommands)."""
    load_config.cache_clear()
