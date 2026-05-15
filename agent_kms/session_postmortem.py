"""Stop hook (low-frequency variant): extract session-level structural failures.

Sister of `session_extract.py`. Where session_extract pulls 3 small actionable
lessons from the most recent 30 turns, this script reads a longer slice of the
transcript and tries to surface **workflow / structural failure modes** that
are only visible at session scope:

  - Stop hook bypassed because magic header was not emitted (today's case)
  - Visual gate skipped under a misleading rationale
  - Scope misinterpretation that propagated through 5 sub-tasks
  - Subagent overconfidence that was not double-checked

These rarely fit the per-turn signal phrase gate, so we run separately with
different prompt focus, higher confidence floor, and a stricter dedup
threshold (since postmortems are inherently broader and easier to paraphrase).

Wired by `.claude/hooks/extract-session-postmortem.sh`. Same Qdrant collection
(`migration_knowledge`) but distinct `source_type=session_postmortem` so
retrieve_for_planning can rank or filter as appropriate.
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
from .session_extract import (
    DEDUP_COS_THRESHOLD,
    FORBIDDEN_RE,
    REJECT_PATTERNS,
    _extract_text_from_content,
)
from .store import (
    COLLECTION,
    encode_passage,
    ensure_collection,
    get_client,
    stable_id,
)

# Higher than session_lesson's 0.85: postmortems are bigger claims and a
# false-positive lesson is more damaging downstream.
POSTMORTEM_CONFIDENCE_FLOOR = 0.90

# Tighter than session_lesson's 0.95: postmortems are easier to paraphrase
# (multiple observers describe the same workflow failure differently) and
# we genuinely want to dedup near-paraphrases here.
POSTMORTEM_DEDUP_COS_THRESHOLD = 0.93

# Each postmortem run produces at most this many entries — keep them few and
# load-bearing.
MAX_POSTMORTEMS = 2

PROMPT_TEMPLATE = """以下は 1 セッション分の Claude Code transcript の長尺要約です。
**workflow / 構造的な失敗 mode** を抽出してください — 単一 turn の小さな間違い
ではなく、session 全体を通して発生した、安全網を bypass する pattern や、
gate を skip する判断や、scope 取り違えが下流タスクに伝播した経路など、
**将来の別セッションが同じ陥穴に落ちないために知るべき構造**を最大 {max_count} 件、
JSON 配列で出力してください。

形式:
[{{
  "text": "失敗 mode の本文 (3-5 文、再発防止のために必要な具体性を含む)",
  "confidence": 0.0-1.0,
  "category": "hook_bypass" | "scope_drift" | "verification_skip" | "subagent_overtrust" | "other",
  "trigger_signal": "次回これが起きそうな時の検知 phrase / ファイル変更 pattern (任意)"
}}]

採用基準 (3 つ全てを満たすもののみ):
1. **構造性**: 単発ミスではなく、複数 step・複数 hook・複数 task に渡って影響した failure mode
2. **再利用性**: 別の作業対象 (別 UI / 別機能 / 別 PR) でも同じ pattern が再来し得る
3. **具体性**: 関与した hook 名 / settings.json key / instinct 名 / skill 名 / file path を含む

却下対象 (該当があれば一切出力しない):
- ❌ 個別 commit / PR の実装ミス (session_lesson 側の領域)
- ❌ "もっと注意する" の vague な advice
- ❌ session 単発の typo / 鍵打ち間違え
- ❌ プロジェクト内部の評価指標 (Tier1A coverage, MRR, NDCG)
- ❌ 廃棄実験の詳細

forbidden vocab (skeleton/placeholder/defer/minimal/scope-narrow/dummy/仮実装/今回のみ) 禁止。
**confidence は構造性・再利用性・具体性への合致度で評価し、{floor} 未満は出力しないこと**。
該当事象が無ければ空配列 [] を返すこと。

transcript summary:
{transcript_full}

JSON only:"""


def read_transcript_full(path: Path, max_text_chars: int = 80000) -> str:
    """Read transcript with *text-aware* backward accumulation.

    The naive byte-tail strategy fails for postmortems on long sessions because
    JSONL transcripts are dominated by tool_result blobs (screenshot binaries,
    large file reads) that get filtered out by `_extract_text_from_content`.
    A 200KB byte tail of a 3.5MB transcript yielded only 7KB of meaningful
    text in production — covering ~10 minutes instead of the session arc.

    Strategy: parse all lines, extract text per turn, then accumulate
    BACKWARDS from the end until reaching `max_text_chars` of *meaningful
    text* (not bytes). This guarantees we capture N turns of conversation
    even when tool_result content is verbose.
    """
    if not path.exists():
        return ""
    text_turns: list[str] = []
    try:
        # Iterate line-by-line so a multi-GB transcript doesn't blow memory.
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
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
                    text_turns.append(f"--- {role} ---\n{text}")
    except OSError:
        return ""

    # Walk backwards, accumulating until we have max_text_chars of text or run
    # out of turns. The leading turns (often user-supplied scope description)
    # are most prone to being truncated — that's a known trade-off; if needed
    # we can sample N turns from the START as well in a future revision.
    selected: list[str] = []
    total = 0
    for turn in reversed(text_turns):
        if total + len(turn) > max_text_chars and selected:
            break
        selected.append(turn)
        total += len(turn) + 2  # +2 for the joining "\n\n"
    selected.reverse()
    return "\n\n".join(selected)


def extract_postmortems(transcript_full: str, max_chars: int = 100000) -> list[dict]:
    if not transcript_full.strip():
        return []
    if len(transcript_full) > max_chars:
        transcript_full = transcript_full[-max_chars:]
    fast_model = os.environ.get("GEMINI_MODEL_FAST")
    result = llm_generate(
        PROMPT_TEMPLATE.format(
            max_count=MAX_POSTMORTEMS,
            floor=POSTMORTEM_CONFIDENCE_FLOOR,
            transcript_full=transcript_full,
        ),
        max_tokens=2048,
        temperature=0.0,
        json_mode=True,
        gemini_model=fast_model,
    )
    print(f"  llm provider used: {result.provider}/{result.model}", file=sys.stderr)
    text = (result.text or "").strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        print("  llm output was not valid JSON; skipping", file=sys.stderr)
        return []
    if not isinstance(items, list):
        return []
    return [
        i
        for i in items
        if isinstance(i, dict) and isinstance(i.get("text"), str) and i["text"].strip()
    ][:MAX_POSTMORTEMS]


def sanitize_postmortems(items: list[dict]) -> list[dict]:
    out = []
    for i in items:
        text = i.get("text", "")
        if FORBIDDEN_RE.search(text):
            continue
        if REJECT_PATTERNS.search(text):
            continue
        if float(i.get("confidence", 0.0)) < POSTMORTEM_CONFIDENCE_FLOOR:
            continue
        out.append(i)
    return out


def _find_postmortem_duplicate(client, vector: list[float]) -> tuple[bool, str]:
    try:
        result = client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=1,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_type",
                        match=MatchValue(value="session_postmortem"),
                    )
                ]
            ),
            with_payload=True,
        )
        hits = result.points
    except Exception as exc:
        print(f"  dedup search failed ({exc}); proceeding with upsert", file=sys.stderr)
        return False, ""
    if not hits:
        return False, ""
    top = hits[0]
    if top.score is None or top.score < POSTMORTEM_DEDUP_COS_THRESHOLD:
        return False, ""
    matched = (top.payload or {}).get("text", "") if isinstance(top.payload, dict) else ""
    return True, matched[:120]


def upsert_postmortems(items: list[dict], session_id: str) -> int:
    if not items:
        return 0
    ensure_collection()
    client = get_client()
    points = []
    deduped = 0
    for idx, item in enumerate(items):
        text = item["text"].strip()
        confidence = float(item.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        category = str(item.get("category", "other"))[:64]
        trigger = str(item.get("trigger_signal", ""))[:256]
        vector = encode_passage(text)

        is_dup, matched_excerpt = _find_postmortem_duplicate(client, vector)
        if is_dup:
            deduped += 1
            print(
                f"  dedup: skipping postmortem '{text[:60]}...' "
                f"(cos>={POSTMORTEM_DEDUP_COS_THRESHOLD} vs '{matched_excerpt}...')",
                file=sys.stderr,
            )
            continue

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        point_id = stable_id(f"postmortem#{session_id}#{idx}#{digest}")
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": text,
                    "source_type": "session_postmortem",
                    "source_file": f"session:{session_id}",
                    "confidence": confidence,
                    "category": category,
                    "trigger_signal": trigger,
                },
            )
        )
    if deduped:
        print(f"  dedup: {deduped} postmortem(s) suppressed", file=sys.stderr)
    if not points:
        return 0
    client.upsert(COLLECTION, points=points, wait=True)
    return len(points)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract session-level structural failures → Qdrant")
    parser.add_argument("--transcript", required=True, help="Path to JSONL transcript")
    parser.add_argument("--session-id", default="", help="Session id")
    args = parser.parse_args()

    if not llm_available():
        print(
            "no LLM provider available (set GEMINI_API_KEY or ANTHROPIC_API_KEY "
            "in tools/rag/.env) -- skipping",
            file=sys.stderr,
        )
        return 0

    session_id = args.session_id or uuid.uuid4().hex[:8]
    full = read_transcript_full(Path(args.transcript))
    if not full:
        print("(no transcript content)", file=sys.stderr)
        return 0

    print(f"transcript chars: {len(full)} (postmortem mode)", file=sys.stderr)
    items = extract_postmortems(full)
    print(f"extracted: {len(items)} postmortem(s)", file=sys.stderr)

    sanitized = sanitize_postmortems(items)
    dropped = len(items) - len(sanitized)
    if dropped:
        print(f"sanitize: dropped {dropped} (forbidden vocab / low confidence)", file=sys.stderr)

    n = upsert_postmortems(sanitized, session_id)
    print(
        f"upserted {n} session_postmortem point(s) (session_id={session_id})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
