"""Retrieval effectiveness summary for one Claude Code session.

Reads two artefacts produced during the session:

  1. ``~/.claude/logs/agent-kms-retrieve.jsonl`` — one JSONL record per
     ``UserPromptSubmit`` hook firing (written by ``auto-rag-retrieve.sh``).
     Each record carries: ``ts``, ``session_id``, ``query``, ``threshold``,
     and a list of ``hits`` (source / heading / score / source_type).
  2. The Claude Code session transcript JSONL.

For each retrieved chunk it asks a single question — *did the assistant
appear to reference this chunk in a later turn?* — using a deliberately
crude substring heuristic over the chunk's heading and source filename
stem. The answer is printed (USED / UNUSED) so the operator can see at
a glance which surfaced knowledge was wasted context vs. genuinely useful.

The heuristic is intentionally simple:

  - false positives are expected when chunks have generic headings;
  - false negatives are expected when the assistant paraphrases instead
    of quoting; the goal is a low-cost feedback signal, not an oracle.

Invoked from the Stop hook (``extract-session-lessons.sh``) so users see a
summary alongside the lesson-extraction output; also available standalone
via ``agent-kms effectiveness --transcript <path>``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

RETRIEVE_LOG = Path.home() / ".claude" / "logs" / "agent-kms-retrieve.jsonl"

# Substring matching ignores case + collapses internal whitespace so
# "Layer 1: Intent" and "layer 1 : intent" both count as matches.
_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WS_RE.sub(" ", s.lower()).strip()


def load_retrieve_events(session_id: str, log_path: Path = RETRIEVE_LOG) -> list[dict]:
    """Return retrieve-log entries filtered to ``session_id``.

    Lines that fail JSON parse are skipped silently — the log is append-only
    from a shell hook so partial writes are possible.
    """
    if not log_path.exists():
        return []
    events: list[dict] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("session_id") == session_id:
            events.append(ev)
    return events


def parse_assistant_texts(transcript_path: Path) -> list[str]:
    """Return the text content of every assistant turn in order.

    Robust to both the flat ``{"role": "assistant", "content": ...}`` and the
    nested ``{"message": {"role": ..., "content": ...}}`` formats Claude Code
    has used over time, and to ``content`` being either a string or a list of
    blocks with ``text`` fields.
    """
    if not transcript_path.exists():
        return []
    out: list[str] = []
    for line in transcript_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    parts.append(c["text"])
            text = "\n".join(parts)
        else:
            text = ""
        if text.strip():
            out.append(text)
    return out


def hit_used(hit: dict, assistant_texts: list[str]) -> tuple[bool, str]:
    """Heuristic: did the assistant reference this retrieved chunk?

    Probes (in order):
      1. The chunk's heading (≥ 5 visible chars after normalisation).
      2. The source file's basename stem (≥ 5 chars, e.g. ``pr-review``).

    The first probe that appears as a substring in ANY assistant turn wins.
    Returns ``(used, matched_phrase)``; ``matched_phrase`` is empty on miss.
    """
    heading = (hit.get("heading") or "").strip()
    source = hit.get("source") or ""
    stem = Path(source).stem if source else ""

    candidates: list[str] = []
    if heading and len(_normalize(heading)) >= 5:
        candidates.append(heading)
    if stem and len(stem) >= 5 and stem not in candidates:
        candidates.append(stem)

    for txt in assistant_texts:
        ntxt = _normalize(txt)
        for cand in candidates:
            if _normalize(cand) in ntxt:
                return True, cand
    return False, ""


def report(
    session_id: str,
    transcript_path: Path,
    log_path: Path = RETRIEVE_LOG,
    out=None,
) -> dict:
    """Print effectiveness summary; return a small dict for testing.

    ``out`` defaults to whatever ``sys.stderr`` is at call time. We resolve
    it lazily (not as a default-arg expression) so pytest's ``capsys``
    fixture — which swaps ``sys.stderr`` after import — captures the output.
    """
    if out is None:
        out = sys.stderr
    events = load_retrieve_events(session_id, log_path)
    header = f"=== retrieval effectiveness (session {session_id[:8]}) ==="
    print(header, file=out)

    if not events:
        print("  (no retrieve events for this session in log)", file=out)
        return {"events": 0, "hits": 0, "used": 0}

    assistant_texts = parse_assistant_texts(transcript_path)

    used: list[tuple[dict, str, str]] = []
    unused: list[tuple[dict, str]] = []
    total_hits = 0
    for ev in events:
        q = ev.get("query", "")
        for hit in ev.get("hits", []):
            total_hits += 1
            is_used, match = hit_used(hit, assistant_texts)
            if is_used:
                used.append((hit, q, match))
            else:
                unused.append((hit, q))

    pct = (len(used) * 100 // total_hits) if total_hits else 0
    print(
        f"  queries: {len(events)}  hits: {total_hits}  "
        f"used: {len(used)} ({pct}%)",
        file=out,
    )

    def _fmt_hit(hit: dict) -> str:
        src = Path(hit.get("source", "")).name or "?"
        heading = (hit.get("heading") or "").strip()[:60]
        score = float(hit.get("score") or 0.0)
        return f"{src}#{heading} (score={score:.3f})"

    if used:
        print("  USED:", file=out)
        for hit, q, match in used:
            print(
                f"    ✓ {_fmt_hit(hit)}  q={q[:40]!r}  match={match[:40]!r}",
                file=out,
            )
    if unused:
        print("  UNUSED:", file=out)
        for hit, q in unused:
            print(f"    ✗ {_fmt_hit(hit)}  q={q[:40]!r}", file=out)

    return {"events": len(events), "hits": total_hits, "used": len(used)}


def main() -> int:
    ap = argparse.ArgumentParser(description="agent-kms retrieval effectiveness")
    ap.add_argument("--transcript", required=True, type=Path)
    ap.add_argument(
        "--session-id",
        default=None,
        help="Defaults to transcript filename stem.",
    )
    ap.add_argument(
        "--log",
        default=str(RETRIEVE_LOG),
        help="Path to the retrieve JSONL log.",
    )
    args = ap.parse_args()

    sid = args.session_id or args.transcript.stem
    report(sid, args.transcript, log_path=Path(args.log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
