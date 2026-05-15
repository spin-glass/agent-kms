"""Stop hook: extract migration lessons from a session transcript -> Qdrant.

Invoked by `.claude/hooks/extract-session-lessons.sh` after each Stop event.
Uses `llm.generate()` so Gemini 2.5 Flash (free tier) is tried first and Haiku
4.5 is the fallback. Failures are non-fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from pathlib import Path

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)

from .llm import generate as llm_generate
from .llm import is_available as llm_available
from .store import (
    COLLECTION,
    encode_passage,
    ensure_collection,
    get_client,
    stable_id,
)

# Cosine similarity threshold above which a candidate lesson is considered a
# duplicate and skipped at upsert time. Tuned empirically against the live
# migration_knowledge collection (multilingual-e5 base encoder):
#
#   exact text duplicate     → 1.0000  (caught — and the DB already has
#                                       multiple, this is the bug we solve)
#   same-meaning paraphrase  → 0.9197  (NOT caught — too close to topical
#                                       neighbors to disambiguate reliably)
#   topical-but-distinct     → 0.9191  (NOT caught, correctly)
#
# 0.95 is the safe band: it eliminates the duplicate-upsert pathology without
# risking false-positive suppression of legitimate near-paraphrases. Tighten to
# 0.93 once we have a sample of clearly-paraphrased lessons that scored above
# topical neighbors (currently encoder cannot reliably split them).
DEDUP_COS_THRESHOLD = 0.95

def _load_prompt(name: str, fallback: str) -> str:
    """Load preset prompt; fallback to embedded default."""
    try:
        from .config import load_prompt
        return load_prompt(name)
    except Exception:
        return fallback


_FALLBACK_LESSON_PROMPT = """以下の session transcript の末尾から、
**別 worktree / 別タスクで作業する将来のセッション** が知っていれば
防げた slack を、最大 3 件、JSON 配列で出力してください。

形式: [{{"text": "知見本文 (1-3 文)", "confidence": 0.0-1.0}}]

採用基準 (3 つ全てを満たすもののみ):
1. **再利用性**: 別の作業対象 (別 UI / 別機能 / 別 bug) でも適用される
2. **具体性**: literal 数値, API 名, ファイル種別, 誤った前提など concrete を含む
3. **発生根拠**: session 内で actual 失敗 / 訂正 / 発見を反映 (proposed/予定 は NG)

却下対象 (該当があれば一切出力しない):
- ❌ git/PR/CI/branch の手順 (CLAUDE.md / Constitution に既収載)
- ❌ "X を確認する / Y に気をつける" の vague な advice
- ❌ session 単発のミス (鍵 prefix 打ち間違え, 一回きりの誤操作)
- ❌ 廃棄された実験の実装詳細 (D2Q / hybrid / 廃止 collection 等)
- ❌ プロジェクト内部の評価指標 (Tier1A coverage, MRR, NDCG, ranking 順位)
- ❌ 一回限りの設計判断 (collection 名切替, ファイル削除/移動)
- ❌ 自セッション固有の作業流れ (実装途中のメモ, 次やる予定)

forbidden vocab (skeleton/placeholder/defer/minimal/scope-narrow/dummy/仮実装/今回のみ) 禁止。
**confidence は採用基準への合致度で評価し、0.85 未満は出力しないこと**。
該当事象が無ければ空配列 [] を返すこと。

transcript:
{transcript_tail}

JSON only:"""

CONFIDENCE_FLOOR = 0.85

# Second-pass prompt: extract domain-specific anti-patterns from the
# session. Results are upserted with source_type=anti_pattern +
# severity=critical so retrieve boosts them above default lessons. The
# text below is the language-neutral fallback used when no preset prompt
# is configured — projects with a stable "convention vs. wrong impl vs.
# correct impl" vocabulary should override this via the preset.
_FALLBACK_ANTI_PATTERN_PROMPT = """From the tail of the following session transcript,
extract up to **3 anti-patterns** that were observed — implementations that
violated the project's stated conventions or specifications. Output a JSON
array only.

Format: [{{"text": "anti-pattern body (3-5 sentences covering: the
convention, the wrong implementation, and the correct one)",
        "title": "short title (10-30 chars)",
        "confidence": 0.0-1.0}}]

Selection criteria (ALL must hold):
1. **Observed** — surfaced in THIS session (not hypothetical / proposed).
2. **Specific** — names concrete functions / APIs / numbers / file lines.
3. **Transferable** — describes a structural pattern that would apply to a
   different target in the same project.

Reject:
- ❌ General principles ("verify assets", "write tests") — those go in
  the lesson extractor.
- ❌ One-off typos / single key-mistypes / git ops.
- ❌ Proposed-state TODOs.

**Output language**: Match the transcript's primary natural language. If
the transcript is mostly Japanese, write each ``text`` and ``title`` in
natural, idiomatic Japanese. Code identifiers, file paths, and English
technical terms stay in their original form.

Confidence < 0.85 must not be output. Return ``[]`` if no new anti-pattern
qualifies.

transcript:
{transcript_tail}

JSON only:"""

# Reusability rejection patterns: matches lessons that look like one-off
# implementation narration even if the LLM ignored the prompt instructions.
REJECT_PATTERNS = re.compile(
    r"(^|\W)("
    r"PR (?:が|は)?マージ"
    r"|CI チェック(?:が|の)?完了"
    r"|migration_knowledge_(?:d2q|l2|l1|hybrid)"
    r"|Tier1A|MRR|NDCG"
    r"|D2Q ingest|hybrid retrieve"
    r"|rank ?[>＞]\s*\d+"
    r"|leading ?R"
    r")",
    re.IGNORECASE,
)

FORBIDDEN_RE = re.compile(
    r"skeleton|placeholder|defer|minimal|scope-narrow|narrow scope|dummy|"
    r"仮実装|今回のみ|スコープ縮小",
    re.IGNORECASE,
)


def _extract_text_from_content(content) -> str:
    if isinstance(content, list):
        out: list[str] = []
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text" and isinstance(c.get("text"), str):
                out.append(c["text"])
            elif c.get("type") == "tool_use":
                name = c.get("name", "?")
                out.append(f"[tool_use:{name}]")
            elif c.get("type") == "tool_result":
                inner = c.get("content")
                if isinstance(inner, list):
                    out.append(_extract_text_from_content(inner))
                elif isinstance(inner, str):
                    out.append(inner)
        return "\n".join(s for s in out if s)
    if isinstance(content, str):
        return content
    return ""


def read_transcript_tail(path: Path, n_turns: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-n_turns:] if len(lines) > n_turns else lines
    parts: list[str] = []
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
        role = msg.get("role") or obj.get("type") or "?"
        text = _extract_text_from_content(msg.get("content"))
        if text:
            parts.append(f"--- {role} ---\n{text}")
    return "\n\n".join(parts)


def _coerce_to_list_of_dicts(parsed: object) -> list[dict]:
    """Adapter that accepts every common JSON shape a small LLM may emit
    instead of the requested top-level array.

    Smaller open-source models (qwen2.5:7b, llama3.1:8b, etc.) frequently
    fail to follow "return only a JSON array" and instead emit:
      a. ``[{...}, {...}]``                          ← canonical (passthrough)
      b. ``{"text": "...", "confidence": 0.9}``      ← single object
      c. ``{"lessons": [{...}, {...}]}``             ← wrapped under any key
      d. ``{"tips_for_naming_X": [{...}]}``          ← same, arbitrary key

    Returning ``[]`` for these recoverable shapes silently drops legitimate
    extractions. Recover them; the downstream filters (forbidden vocab,
    confidence floor, dedup) still gate quality.

    Args:
        parsed: result of ``json.loads`` — any Python value.

    Returns:
        ``list[dict]`` (possibly empty). Non-dict entries are filtered out.
    """
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        # Shape (b): single lesson object directly. Heuristic: it carries
        # the canonical "text" field. (anti-patterns additionally require
        # "title"; the per-item validation downstream rejects mismatches.)
        if isinstance(parsed.get("text"), str):
            print("  llm output: recovered single-object → 1-element list", file=sys.stderr)
            return [parsed]
        # Shape (c/d): wrapper object — find the first list value and unwrap.
        for k, v in parsed.items():
            if isinstance(v, list):
                print(
                    f"  llm output: recovered wrapper '{k}' → {len(v)}-element list",
                    file=sys.stderr,
                )
                return [x for x in v if isinstance(x, dict)]
    return []


def extract_lessons(transcript_tail: str, max_chars: int = 60000) -> list[dict]:
    if not transcript_tail.strip():
        return []
    if len(transcript_tail) > max_chars:
        transcript_tail = transcript_tail[-max_chars:]
    # Stop hook = fast tier (Flash-Lite). High volume, structured JSON output.
    fast_model = os.environ.get("GEMINI_MODEL_FAST")
    result = llm_generate(
        _load_prompt("session_extract_lessons", _FALLBACK_LESSON_PROMPT).format(
            transcript_tail=transcript_tail
        ),
        max_tokens=1024,
        temperature=0.0,
        json_mode=True,
        gemini_model=fast_model,
    )
    print(f"  llm provider used: {result.provider}/{result.model}", file=sys.stderr)
    text = (result.text or "").strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print("  llm output was not valid JSON; skipping", file=sys.stderr)
        return []
    lessons = _coerce_to_list_of_dicts(parsed)
    return [
        l
        for l in lessons
        if isinstance(l.get("text"), str) and l["text"].strip()
    ]


def sanitize(lessons: list[dict]) -> list[dict]:
    out = []
    for l in lessons:
        text = l.get("text", "")
        if FORBIDDEN_RE.search(text):
            continue
        if REJECT_PATTERNS.search(text):
            continue
        if float(l.get("confidence", 0.0)) < CONFIDENCE_FLOOR:
            continue
        out.append(l)
    return out


def _find_duplicate(client, vector: list[float]) -> tuple[bool, str]:
    """Search migration_knowledge for an existing session_lesson within
    DEDUP_COS_THRESHOLD of the candidate vector. Returns (is_duplicate,
    matched_text_excerpt) — the excerpt is empty when no duplicate found.

    Filter is restricted to source_type='session_lesson' so cross-source
    lessons (e.g. instincts, known_issues) do not block legitimate session
    knowledge that happens to be semantically similar to a documented
    instinct.
    """
    try:
        result = client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=1,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value="session_lesson"),
                    )
                ]
            ),
            with_payload=True,
        )
        hits = result.points
    except Exception as exc:
        # Search failures are non-fatal — fall through to upsert. Better to
        # accept a duplicate than to silently drop a real lesson.
        print(f"  dedup search failed ({exc}); proceeding with upsert", file=sys.stderr)
        return False, ""

    if not hits:
        return False, ""
    top = hits[0]
    if top.score is None or top.score < DEDUP_COS_THRESHOLD:
        return False, ""
    matched = (top.payload or {}).get("text", "") if isinstance(top.payload, dict) else ""
    return True, matched[:120]


def extract_anti_patterns(transcript_tail: str, max_chars: int = 60000) -> list[dict]:
    """Second-pass extraction: convention-violation anti-patterns observed
    in the session.

    Returns list of ``{text, title, confidence}`` entries. Empty on no
    observation or LLM failure.
    """
    if not transcript_tail.strip():
        return []
    if len(transcript_tail) > max_chars:
        transcript_tail = transcript_tail[-max_chars:]
    fast_model = os.environ.get("GEMINI_MODEL_FAST")
    try:
        result = llm_generate(
            _load_prompt(
                "session_extract_anti_patterns", _FALLBACK_ANTI_PATTERN_PROMPT
            ).format(transcript_tail=transcript_tail),
            max_tokens=2048,
            temperature=0.0,
            json_mode=True,
            gemini_model=fast_model,
        )
    except Exception as exc:
        print(f"  anti-pattern extract failed: {exc}", file=sys.stderr)
        return []
    text = (result.text or "").strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print("  anti-pattern LLM output not valid JSON; skipping", file=sys.stderr)
        return []
    items = _coerce_to_list_of_dicts(parsed)
    out = []
    for it in items:
        body = it.get("text", "").strip()
        title = it.get("title", "").strip()
        if not body or not title:
            continue
        if FORBIDDEN_RE.search(body) or REJECT_PATTERNS.search(body):
            continue
        conf = float(it.get("confidence", 0.0))
        if conf < CONFIDENCE_FLOOR:
            continue
        out.append({"text": body, "title": title, "confidence": conf})
    return out


def upsert_anti_patterns(items: list[dict], session_id: str) -> int:
    """Upsert anti-pattern entries as source_type='anti_pattern' / severity='critical'."""
    if not items:
        return 0
    ensure_collection()
    points = []
    for idx, it in enumerate(items):
        text = it["text"]
        title = it["title"]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        point_id = stable_id(f"anti_pattern#{session_id}#{idx}#{digest}")
        points.append(
            PointStruct(
                id=point_id,
                vector=encode_passage(text),
                payload={
                    "text": text,
                    "heading": title,
                    "source_type": "anti_pattern",
                    "source_file": f"session:{session_id}",
                    "confidence": float(it["confidence"]),
                    "severity": "critical",
                    "applicability": "universal",
                },
            )
        )
    get_client().upsert(COLLECTION, points=points, wait=True)
    return len(points)


def upsert_lessons(lessons: list[dict], session_id: str) -> int:
    if not lessons:
        return 0
    ensure_collection()
    client = get_client()
    points = []
    deduped = 0
    for idx, lesson in enumerate(lessons):
        text = lesson["text"].strip()
        confidence = float(lesson.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        vector = encode_passage(text)

        is_dup, matched_excerpt = _find_duplicate(client, vector)
        if is_dup:
            deduped += 1
            print(
                f"  dedup: skipping '{text[:60]}...' (cos>={DEDUP_COS_THRESHOLD} "
                f"vs existing '{matched_excerpt}...')",
                file=sys.stderr,
            )
            continue

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        point_id = stable_id(f"{session_id}#{idx}#{digest}")
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": text,
                    "source_type": "session_lesson",
                    "source_file": f"session:{session_id}",
                    "confidence": confidence,
                },
            )
        )
    if deduped:
        print(f"  dedup: {deduped} lesson(s) suppressed", file=sys.stderr)
    if not points:
        return 0
    client.upsert(COLLECTION, points=points, wait=True)
    return len(points)


def _streak_record(success: bool) -> None:
    """Append 'S' or 'F' to the streak log. When the last STREAK_ALERT_THRESHOLD
    runs are all 'F', fire a macOS notification once and write a debounce
    marker so the same streak does not re-alert until the next 'S' resets it.

    "Failure" here means upserted=0 (LLM failure, all-deduped, all-sanitized,
    or empty extraction). Whether that's a true failure depends on context —
    the alert exists to surface *runs* of zero-output, which usually indicate
    an upstream problem (API quota, prompt regression, dedup over-trigger).
    """
    streak_dir = Path.home() / ".claude" / "logs"
    streak_dir.mkdir(parents=True, exist_ok=True)
    streak_file = streak_dir / "extract-session-lesson-streak.log"
    debounce_file = streak_dir / "extract-session-lesson-streak.alerted"

    char = "S" if success else "F"
    try:
        with streak_file.open("a", encoding="utf-8") as f:
            f.write(char)
        # Trim to last 200 chars so the file does not grow unbounded.
        contents = streak_file.read_text(encoding="utf-8")
        if len(contents) > 200:
            streak_file.write_text(contents[-200:], encoding="utf-8")
            contents = contents[-200:]
    except OSError:
        return

    if success:
        # Successful run resets debounce — next FFFFF can re-alert.
        debounce_file.unlink(missing_ok=True)
        return

    threshold = STREAK_ALERT_THRESHOLD
    if len(contents) < threshold:
        return
    if contents[-threshold:] != "F" * threshold:
        return
    if debounce_file.exists():
        return

    # Fire one notification per uninterrupted F-streak.
    try:
        debounce_file.write_text(contents[-threshold:], encoding="utf-8")
        message = (
            f"agent-kms: {threshold} consecutive zero-extract runs "
            f"(check ~/.claude/logs/agent-kms-stop.log)"
        )
        # osascript is no-op on non-macOS; soft-fail.
        os.system(  # noqa: S605 — message is fully controlled
            f'osascript -e \'display notification "{message}" '
            f'with title "agent-kms streak alert"\' >/dev/null 2>&1'
        )
        print(f"  STREAK ALERT: {threshold} consecutive zero-extracts", file=sys.stderr)
    except OSError:
        pass


# Number of consecutive zero-extract runs that triggers a desktop alert. 5 is
# tuned so transient single-call failures (rate limit blips) do not page the
# operator, while persistent breakage (API key stale, prompt regression,
# dedup over-trigger) surfaces within ~5 sessions.
STREAK_ALERT_THRESHOLD = 5


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract session lessons → Qdrant")
    parser.add_argument("--transcript", required=True, help="Path to JSONL transcript")
    parser.add_argument("--session-id", default="", help="Session id")
    parser.add_argument("--tail-turns", type=int, default=30)
    args = parser.parse_args()

    if not llm_available():
        print(
            "no LLM provider available (set GEMINI_API_KEY or ANTHROPIC_API_KEY "
            "in tools/rag/.env) -- skipping",
            file=sys.stderr,
        )
        return 0

    session_id = args.session_id or uuid.uuid4().hex[:8]
    tail = read_transcript_tail(Path(args.transcript), args.tail_turns)
    if not tail:
        print("(no transcript content)", file=sys.stderr)
        return 0

    print(f"transcript chars: {len(tail)}", file=sys.stderr)
    lessons = extract_lessons(tail)
    print(f"extracted: {len(lessons)} lesson(s)", file=sys.stderr)

    sanitized = sanitize(lessons)
    dropped = len(lessons) - len(sanitized)
    if dropped:
        print(f"sanitize: dropped {dropped} (forbidden vocab)", file=sys.stderr)

    n = upsert_lessons(sanitized, session_id)
    print(f"upserted {n} session_lesson point(s) (session_id={session_id})", file=sys.stderr)
    _streak_record(success=(n > 0))

    # Second pass: extract convention-violation anti-patterns from the
    # session (the "wrong implementation observed vs. correct pattern"
    # variant). Tagged source_type='anti_pattern' / severity='critical'
    # so the retrieve severity boost keeps them above default lessons.
    anti = extract_anti_patterns(tail)
    print(f"extracted: {len(anti)} anti_pattern(s)", file=sys.stderr)
    n_anti = upsert_anti_patterns(anti, session_id)
    if n_anti:
        print(f"upserted {n_anti} anti_pattern point(s) (session_id={session_id})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
