"""Minimal smoke tests — no Qdrant / LLM required."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_import_version():
    import agent_kms
    assert agent_kms.__version__ == "0.1.0"


def test_cli_parser_builds():
    from agent_kms.cli import build_parser
    p = build_parser()
    assert p.prog == "agent-kms"


def test_config_loads_general_preset():
    from agent_kms.config import load_config, reset_cache
    reset_cache()
    cfg = load_config("general")
    assert cfg.preset == "general"
    assert cfg.collection == "agent_knowledge"
    assert any(s.kind == "markdown_h2" for s in cfg.sources)
    assert cfg.score_threshold == 0.93


def test_config_loads_cocos_preset():
    from agent_kms.config import load_config, reset_cache
    reset_cache()
    cfg = load_config("cocos_unity")
    assert cfg.collection == "migration_knowledge"
    assert cfg.file_severity_map.get("docs/cocos-unity-mapping.md") == (
        "critical",
        "universal",
    )


def test_load_prompt_general():
    from agent_kms.config import load_prompt, reset_cache
    reset_cache()
    text = load_prompt("session_extract_lessons", "general")
    assert "{transcript_tail}" in text


def test_chunk_markdown_h2(tmp_path: Path):
    from agent_kms.chunker import chunk_markdown_h2

    md = tmp_path / "x.md"
    md.write_text("# Title\nPreamble.\n\n## Section A\n" + "a" * 200 + "\n\n## Section B\n" + "b" * 200)
    chunks = list(chunk_markdown_h2(tmp_path, min_chars=100))
    headings = [c.heading for c in chunks]
    assert "Section A" in headings
    assert "Section B" in headings


def test_chunk_yaml_per_file(tmp_path: Path):
    from agent_kms.chunker import chunk_yaml_per_file

    y = tmp_path / "thing.yaml"
    y.write_text("name: example\ndomain: workflow\n")
    chunks = list(chunk_yaml_per_file(tmp_path))
    assert len(chunks) == 1
    assert chunks[0].heading == "thing"


def test_stable_id_is_deterministic():
    from agent_kms.store import stable_id
    assert stable_id("foo#bar") == stable_id("foo#bar")
    assert stable_id("foo#bar") != stable_id("foo#baz")


def test_session_extract_fallback_prompt_available():
    """Even without preset prompts, the embedded fallback must work."""
    from agent_kms.session_extract import _FALLBACK_LESSON_PROMPT
    assert "{transcript_tail}" in _FALLBACK_LESSON_PROMPT
