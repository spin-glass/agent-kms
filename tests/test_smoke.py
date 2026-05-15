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


def test_capture_slugify():
    """Slugs must be filesystem-safe + date-prefixed + collision-resistant."""
    from agent_kms.capture import slugify

    # ASCII title → readable slug
    s = slugify("Prisma client import path is wrong")
    assert s.startswith(f"{__import__('datetime').date.today().isoformat()}-")
    assert "prisma-client-import-path-is-wrong" in s

    # Mixed-script title: ASCII tokens are preserved (readable), non-ASCII
    # tokens drop out — but enough ASCII remains so no hash fallback.
    s_mixed = slugify("Prisma クライアントの import 経路")
    assert s_mixed.startswith(f"{__import__('datetime').date.today().isoformat()}-")
    assert "prisma" in s_mixed and "import" in s_mixed

    # All-Japanese title → ASCII part empty, must fall back to hash slug.
    s_jp = slugify("クライアントの設定方法を確認したい")
    assert s_jp.startswith(f"{__import__('datetime').date.today().isoformat()}-")
    assert "lesson-" in s_jp

    # Two different titles produce different slugs
    assert slugify("Foo bar baz qux") != slugify("Quux corge grault")


def test_capture_render_includes_frontmatter():
    """The rendered file must contain machine-readable frontmatter + H1 + body."""
    from agent_kms.capture import render

    out = render(
        title="PR feedback X",
        body="Some markdown body.",
        severity="critical",
        applicability="universal",
        tags=["foo", "bar"],
        source_pr="https://github.com/o/r/pull/123",
    )
    assert "---" in out
    assert "severity: critical" in out
    assert "applicability: universal" in out
    assert "tags: [foo, bar]" in out
    assert "source_pr: https://github.com/o/r/pull/123" in out
    assert "# PR feedback X" in out
    assert "Some markdown body." in out


def test_capture_write_does_not_overwrite_silently(tmp_path):
    """Two captures with the same slug on the same day must not overwrite."""
    from agent_kms.capture import write_file

    p1 = write_file(tmp_path, "2026-05-15-x", "content A", force=False)
    p2 = write_file(tmp_path, "2026-05-15-x", "content B", force=False)
    assert p1 != p2  # second one got a hash suffix
    assert p1.read_text() == "content A"
    assert p2.read_text() == "content B"


def test_effectiveness_hit_used_heuristic():
    """The substring heuristic must:
      - match when the assistant quotes the heading (any case / whitespace)
      - match on the filename stem as a fallback
      - skip too-short heading + stem (heuristic ignores them)
      - miss when the assistant uses neither
    """
    from agent_kms.effectiveness import hit_used

    asst = [
        "Looking at pr-review.md, the PR Review framework is what we want.",
        "Earlier I checked the layer-1 intent step.",
    ]

    # Match on heading (whitespace + case insensitive)
    assert hit_used(
        {"source": "/a/b/pr-review.md", "heading": "PR REVIEW"}, asst
    )[0] is True

    # Match on filename stem when heading is too short / generic
    assert hit_used(
        {"source": "/a/b/pr-review.md", "heading": "x"}, asst
    )[0] is True

    # Miss — neither heading nor stem are referenced anywhere
    assert hit_used(
        {"source": "/a/b/unrelated.md", "heading": "Something Different"},
        asst,
    ) == (False, "")

    # Too-short heading AND too-short stem → no candidates, miss
    assert hit_used(
        {"source": "/a/b/x.md", "heading": "lo"}, asst
    ) == (False, "")


def test_effectiveness_report_with_no_log(tmp_path, capsys):
    """When no retrieve log exists, report prints a friendly note and exits."""
    from agent_kms.effectiveness import report

    transcript = tmp_path / "session-abc.jsonl"
    transcript.write_text("", encoding="utf-8")
    missing_log = tmp_path / "no-such.jsonl"

    result = report("abc12345-fake-session-id", transcript, log_path=missing_log)
    captured = capsys.readouterr()
    assert "no retrieve events" in captured.err
    assert result == {"events": 0, "hits": 0, "used": 0}


def test_coerce_to_list_of_dicts():
    """Lenient JSON-shape adapter: small LLMs often wrap or unwrap the
    requested array. The adapter recovers common shapes so legitimate
    lessons are not silently dropped.
    """
    from agent_kms.session_extract import _coerce_to_list_of_dicts

    # (a) canonical: list passthrough, non-dict filtered
    assert _coerce_to_list_of_dicts([{"text": "a"}, {"text": "b"}]) == [
        {"text": "a"},
        {"text": "b"},
    ]
    assert _coerce_to_list_of_dicts([{"text": "a"}, "garbage", 42]) == [{"text": "a"}]

    # (b) single object with "text" → 1-element list
    assert _coerce_to_list_of_dicts({"text": "lone", "confidence": 0.9}) == [
        {"text": "lone", "confidence": 0.9}
    ]

    # (c) wrapped under any key (qwen2.5:7b typical: "lessons", "tips_for_X")
    assert _coerce_to_list_of_dicts({"lessons": [{"text": "x"}, {"text": "y"}]}) == [
        {"text": "x"},
        {"text": "y"},
    ]
    assert _coerce_to_list_of_dicts({"tips_for_naming_X": [{"text": "z"}]}) == [
        {"text": "z"}
    ]

    # (d) unrecognised shapes → empty (parsers reject quietly)
    assert _coerce_to_list_of_dicts(None) == []
    assert _coerce_to_list_of_dicts("just a string") == []
    assert _coerce_to_list_of_dicts(42) == []
    assert _coerce_to_list_of_dicts({"no_text_no_list": "scalar"}) == []


def test_resolve_collection_precedence(monkeypatch, tmp_path):
    """Collection resolution: env > project kms.toml > default.

    Regression for a bug where ``ensure_collection`` ignored ``kms.toml``
    ``[store].collection`` (only env was honoured), causing ``upsert``
    to write to a different collection than the one ``ensure_collection``
    created.
    """
    from agent_kms import config as cfg_mod
    from agent_kms.store import _resolve_collection

    # 1. default when nothing is set
    monkeypatch.delenv("AGENT_KMS_COLLECTION", raising=False)
    monkeypatch.delenv("AGENT_KMS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg_mod.reset_cache()
    assert _resolve_collection() == "agent_knowledge"

    # 2. kms.toml override (no env) — this is the case the bug missed
    project = tmp_path / ".agent-kms"
    project.mkdir()
    (project / "kms.toml").write_text(
        '[store]\ncollection = "from_toml"\n', encoding="utf-8"
    )
    cfg_mod.reset_cache()
    assert _resolve_collection() == "from_toml"

    # 3. env wins over kms.toml
    monkeypatch.setenv("AGENT_KMS_COLLECTION", "from_env")
    cfg_mod.reset_cache()
    assert _resolve_collection() == "from_env"

    # cleanup so other tests get a fresh cache
    cfg_mod.reset_cache()


def test_construct_recognizes_ollama(monkeypatch):
    """_construct must route 'ollama' to _OllamaProvider, even if daemon is down.

    We point OLLAMA_URL at an unreachable host and expect a RuntimeError whose
    message is the liveness-probe failure — NOT the generic 'unknown provider'.
    """
    from agent_kms import llm

    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:1")
    with pytest.raises(RuntimeError) as ei:
        llm._construct("ollama")
    msg = str(ei.value)
    assert "unknown provider" not in msg
    assert "Ollama unreachable" in msg or "ollama" in msg.lower()


def _ollama_reachable(url: str = "http://localhost:11434") -> bool:
    """Probe Ollama's /api/tags; return True iff it responds within 1s."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=1) as r:
            r.read()
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama daemon not reachable")
def test_ollama_end_to_end_generate(monkeypatch):
    """Real E2E: when Ollama daemon is up, agent_kms.llm.generate() must
    successfully route through the Ollama provider and return non-empty text.

    Skipped automatically when the daemon is down so the test stays a true
    smoke test (no infrastructure required for the rest of the suite)."""
    import os

    from agent_kms.llm import chain_summary, generate

    # Force Ollama-only chain regardless of how the user has env set
    monkeypatch.setenv("RAG_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"))
    monkeypatch.delenv("RAG_PROVIDER_FALLBACK", raising=False)

    summary = chain_summary()
    assert summary.startswith("ollama:"), f"expected ollama-only chain, got: {summary}"

    result = generate(
        "Reply with exactly: pong",
        max_tokens=16,
        temperature=0.0,
    )
    assert result.provider == "ollama"
    assert result.text.strip(), "expected non-empty text from Ollama"
