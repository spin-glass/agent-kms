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
    assert cfg.score_threshold == 0.83


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


def test_chunk_markdown_h2_frontmatter(tmp_path: Path):
    """YAML frontmatter on a markdown file must override per-source defaults
    and propagate optional fields (``source_pr``, ``captured_at``, ``tags``)
    onto every chunk derived from the file.
    """
    from agent_kms.chunker import chunk_markdown_h2

    md = tmp_path / "pr-lesson.md"
    md.write_text(
        "---\n"
        "captured_at: 2026-05-15T16:00:00\n"
        "severity: critical\n"
        "applicability: universal\n"
        "tags: [prisma, import]\n"
        "source_pr: https://github.com/o/r/pull/123\n"
        "---\n\n"
        "# Title\n\n"
        "## Section A\n" + "a" * 200 + "\n\n"
        "## Section B\n" + "b" * 200 + "\n"
    )

    chunks = list(
        chunk_markdown_h2(
            tmp_path,
            min_chars=100,
            # These defaults must be overridden by the frontmatter values.
            default_severity="default",
            default_applicability="topic-specific",
        )
    )
    assert len(chunks) >= 2
    for c in chunks:
        assert c.severity == "critical"             # frontmatter wins over default
        assert c.applicability == "universal"
        assert "prisma" in c.tags and "import" in c.tags
        assert c.source_pr == "https://github.com/o/r/pull/123"
        assert c.captured_at == "2026-05-15T16:00:00"


def test_chunk_markdown_h2_no_frontmatter_unchanged(tmp_path: Path):
    """Legacy markdown without a frontmatter block must round-trip with
    per-source defaults intact and the new optional fields empty.
    """
    from agent_kms.chunker import chunk_markdown_h2

    md = tmp_path / "legacy.md"
    md.write_text("# Title\n\n## Section\n" + "x" * 200)
    chunks = list(chunk_markdown_h2(tmp_path, min_chars=100, default_severity="high"))
    assert chunks
    for c in chunks:
        assert c.severity == "high"        # default preserved
        assert c.source_pr == ""           # empty optional field
        assert c.captured_at == ""
        assert c.tags == []


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


def test_capture_pr_parse_ref():
    """All four PR-ref forms parse to the same {owner,repo,number} shape."""
    from agent_kms.capture_pr import parse_pr_ref

    # Canonical URL
    p = parse_pr_ref("https://github.com/japan-ai-inc/japan-ai/pull/1234")
    assert p["owner"] == "japan-ai-inc"
    assert p["repo"] == "japan-ai"
    assert p["number"] == 1234

    # owner/repo#N short form
    p = parse_pr_ref("japan-ai-inc/japan-ai#42")
    assert p["owner"] == "japan-ai-inc"
    assert p["repo"] == "japan-ai"
    assert p["number"] == 42

    # bare #N — number resolved later by `gh` from the cwd's repo
    p = parse_pr_ref("#7")
    assert p["number"] == 7
    assert "owner" not in p

    # bare digits — same idea
    p = parse_pr_ref("99")
    assert p["number"] == 99
    assert "owner" not in p

    # Malformed → ValueError
    import pytest as _p

    with _p.raises(ValueError):
        parse_pr_ref("not-a-ref")


def test_capture_pr_normalise_items():
    """A synthetic gh-shaped payload normalises into the expected list shape
    with stable codes (r1/c1/i1...) so user selection works deterministically.
    """
    from agent_kms.capture_pr import normalise_items

    pr = {
        "number": 1,
        "title": "test",
        "url": "https://github.com/o/r/pull/1",
        "reviews": [
            {"state": "CHANGES_REQUESTED", "body": "fix DB transaction",
             "author": {"login": "alice"}, "url": "https://github.com/o/r/pull/1#r-1"},
            {"state": "COMMENTED", "body": "", "author": {"login": "bob"}},  # empty → dropped
        ],
        "comments": [
            {"body": "general nit", "author": {"login": "carol"},
             "url": "https://github.com/o/r/pull/1#c-1"},
        ],
    }
    inline = [
        {"body": "use core-prisma", "path": "src/db.ts", "line": 12,
         "diff_hunk": "@@\n-import ...\n+import ...",
         "user": {"login": "alice"},
         "html_url": "https://github.com/o/r/pull/1#discussion-1"},
    ]

    items = normalise_items(pr, inline)
    codes = [x["code"] for x in items]
    assert codes == ["r1", "c1", "i1"]  # empty review dropped, others present
    assert items[0]["state"] == "CHANGES_REQUESTED"
    assert items[2]["path"] == "src/db.ts"
    assert items[2]["line"] == 12


def test_capture_pr_build_capture_args_severity_mapping():
    """Severity must follow the CHANGES_REQUESTED→critical, others→high rule
    so the retrieve boost can lift the most-painful feedback to the top.
    """
    from agent_kms.capture_pr import build_capture_args

    pr = {"number": 7, "url": "https://github.com/o/r/pull/7"}
    review_cr = {"code": "r1", "kind": "review", "state": "CHANGES_REQUESTED",
                 "body": "transactionで囲んで。", "author": "alice", "url": "u",
                 "path": None, "line": None, "diff_hunk": None}
    review_comment = {**review_cr, "state": "COMMENTED"}
    inline = {"code": "i1", "kind": "inline", "state": "INLINE",
              "body": "core-prisma を使う。", "author": "bob",
              "path": "src/db.ts", "line": 12, "diff_hunk": "@@\n-x\n+y",
              "url": "u"}

    _, _, sev_cr, tags_cr = build_capture_args(review_cr, pr, [])
    assert sev_cr == "critical"
    assert "pr-review" in tags_cr

    _, _, sev_co, _ = build_capture_args(review_comment, pr, [])
    assert sev_co == "high"

    title_in, body_in, sev_in, tags_in = build_capture_args(inline, pr, [])
    assert sev_in == "high"
    assert "inline" in tags_in
    assert "src/db.ts" in title_in
    assert "src/db.ts:12" in body_in  # file:line context inlined


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


# ──────────────────────────────────────────────────────────────────────
# calibrate-threshold (no Qdrant — uses an in-process retrieve stub)
# ──────────────────────────────────────────────────────────────────────


def test_calibrate_precision_recall_f1_basic():
    from agent_kms.calibrate import precision_recall_f1

    retrieved = [
        {"source": "/abs/a.md", "heading": "H1"},   # hit (basename + heading)
        {"source": "b.md", "heading": "X"},          # miss
    ]
    gold = {("a.md", "H1"), ("c.md", "H2")}          # 2 expected, 1 hit
    p, r, f1 = precision_recall_f1(retrieved, gold)
    assert p == 0.5
    assert r == 0.5
    assert round(f1, 4) == 0.5


def test_calibrate_gold_heading_optional_matches_any():
    from agent_kms.calibrate import _gold_set, precision_recall_f1

    gold = _gold_set({"gold": [{"source": "foo.yaml"}]})  # no heading
    retrieved = [{"source": "/x/foo.yaml", "heading": "anything"}]
    p, r, f1 = precision_recall_f1(retrieved, gold)
    assert p == 1.0 and r == 1.0 and f1 == 1.0


def test_calibrate_sweep_picks_best_f1(tmp_path):
    from agent_kms.calibrate import frange, run

    # Stub retrieve_fn: at low T returns both relevant + noise; at high T
    # filters everything out. The F1-optimal threshold is in the middle.
    relevant = {"source": "a.md", "heading": "H1", "score": 0.85}
    noise = {"source": "noise.md", "heading": "N", "score": 0.80}

    def fake_retrieve(query, score_threshold):
        return [c for c in (relevant, noise) if c["score"] >= score_threshold]

    queries_yaml = tmp_path / "q.yaml"
    queries_yaml.write_text(
        "- id: only\n"
        "  query: q\n"
        "  gold:\n"
        "    - source: a.md\n"
        "      heading: H1\n",
        encoding="utf-8",
    )
    out = run(queries_yaml, 0.78, 0.90, 0.01, retrieve_fn=fake_retrieve)
    # Best F1 marker should land at 0.81–0.85 band (relevant in, noise out)
    assert "best F1" in out
    best_lines = [
        ln for ln in out.splitlines() if "best F1" in ln
    ]
    assert best_lines
    # All "best F1" lines should have threshold strictly above noise (0.80)
    # and at or below relevant (0.85)
    for ln in best_lines:
        t = float(ln.split()[0])
        assert 0.81 <= t <= 0.85, ln

    # Sanity: frange inclusive at both ends, no float drift.
    rng = frange(0.78, 0.94, 0.01)
    assert rng[0] == 0.78 and rng[-1] == 0.94


def test_calibrate_load_queries_validates_schema(tmp_path):
    from agent_kms.calibrate import load_queries

    bad = tmp_path / "bad.yaml"
    bad.write_text("- query: missing gold\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or empty 'gold'"):
        load_queries(bad)
