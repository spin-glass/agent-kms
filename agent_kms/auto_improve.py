"""Detect retrieval gaps from a session transcript and auto-improve instinct YAMLs.

Called from the Stop hook `auto-improve-rag.sh` as a detached background
process so the user's main session is never blocked.

Pipeline
========
1. Parse the transcript JSONL for user messages.
2. For each message, run `retrieve(prompt, score_threshold=0.85)` and record
   the source files of the top-10 chunks.
3. Apply GAP_RULES — regex over user prompts that map symptom phrases to
   "expected instinct source files". When the expected file is absent from
   the top-10 (= retrieval miss), record a gap.
4. For each gap, extract Japanese symptom phrases from the user prompt and
   append them to a `# auto-improve symptom keywords:` comment block in the
   instinct YAML frontmatter. The trigger field is NEVER overwritten — only
   the comment block grows, so manual curation stays the source of truth.
5. Re-invoke `agent_kms.ingest` to refresh Qdrant.

Idempotency
-----------
The keyword-append step is dedup-aware: phrases already present in the file
are skipped. Re-running on the same transcript yields zero diff.

Limits
------
- Append-only edits: existing trigger / body never overwritten.
- 64 KB transcript window scan (last few turns; matches extract-session-lessons).
- Max 5 gap rules fired per session (avoids runaway append loops).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator


def _resolve_repo_root() -> Path:
    """Resolve the project root (env > config > cwd) lazily."""
    import os
    from .config import load_config

    env = os.environ.get("AGENT_KMS_KNOWLEDGE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    try:
        return load_config().knowledge_root
    except Exception:
        return Path.cwd().resolve()


REPO_ROOT = _resolve_repo_root()
INSTINCTS_DIR = REPO_ROOT / "data" / "instincts"
LOG_DIR = Path.home() / ".claude" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "agent-kms-auto-improve.log"


# ---------------------------------------------------------------------------
# Gap rules
# ---------------------------------------------------------------------------
# Each rule pairs a regex (over user prompt text) with the instinct file that
# SHOULD have surfaced at the moment that prompt was sent. If the file is not
# in the retrieval top-10 for the prompt, the rule fires and the matching
# symptom phrases from the prompt are appended to the file's keyword block.
#
# regex MUST match Japanese symptom phrasing the user is likely to type when
# the corresponding pitfall occurs. Add new rules only when a session has
# concrete evidence of a missed retrieval; keep the table small.

def load_gap_rules() -> list[dict]:
    """Load gap rules from the active project's ``kms.toml``.

    Each rule has shape::

        {"id": str, "regex": re.Pattern, "expected_source": str}

    Source TOML::

        [[auto_improve.rules]]
        id = "example-rule"
        regex = "(symptom phrase 1|symptom phrase 2)"
        expected_source = "data/instincts/example.yaml"

    Returns ``[]`` when no rules are configured — auto-improve becomes a
    no-op rather than firing built-in rules that belong to one specific
    project's symptom vocabulary.
    """
    try:
        from .config import load_config

        cfg = load_config()
        rules = cfg.raw.get("auto_improve", {}).get("rules", [])
        out: list[dict] = []
        for r in rules:
            try:
                out.append(
                    {
                        "id": r["id"],
                        "regex": re.compile(r["regex"], re.IGNORECASE),
                        "expected_source": r["expected_source"],
                    }
                )
            except (KeyError, re.error):
                continue
        return out
    except Exception:
        return []

MAX_GAPS_PER_SESSION = 5
RETRIEVE_TOP_N = 10
RETRIEVE_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def parse_user_messages(transcript_path: Path, max_records: int = 50) -> list[str]:
    """Return up to `max_records` most-recent user-typed prompts from the transcript.

    Claude Code transcripts mix two record shapes under `type:"user"`:
      (a) user-typed prompt: `message.content` is a STRING (the actual prompt)
      (b) tool-result feedback: `message.content` is a LIST of content blocks
          (tool_result / text), generated automatically when an assistant tool
          call completes
    Only (a) is useful for symptom-keyword detection. Shape (b) records can
    each be 100s of KB (large tool outputs), so a fixed-byte tail window often
    contains zero (a) records. Streaming line-by-line is cheap because the
    JSON parser only retains one line at a time.

    Iterates the entire file but stops accumulating once `max_records` recent
    string-content user records are collected (reverse-scan via deque tail).
    """
    if not transcript_path.exists():
        return []
    from collections import deque
    recent: deque[str] = deque(maxlen=max_records)
    try:
        with transcript_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "user":
                    continue
                msg = obj.get("message")
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                # Shape (a) only — string content == user-typed prompt
                if isinstance(content, str) and len(content) >= 5:
                    recent.append(content)
    except OSError:
        return []
    return list(recent)


def run_retrieve(query: str) -> list[str]:
    """Return source files of top-N chunks from retrieve(). Empty on failure."""
    try:
        from agent_kms.retrieve import retrieve
    except Exception as exc:
        _log(f"retrieve import failed: {exc}")
        return []
    try:
        res = retrieve(query, score_threshold=RETRIEVE_THRESHOLD)
    except Exception as exc:
        _log(f"retrieve call failed: {exc}")
        return []
    return [r.get("source", "") for r in res[:RETRIEVE_TOP_N]]


def extract_symptom_phrases(prompt: str, regex: re.Pattern) -> list[str]:
    """Return the unique regex matches found in the prompt (capped at 5).

    Used to seed the keyword block with literal phrases that the user just
    typed — those are the highest-value embedding bias signals.
    """
    matches = regex.findall(prompt)
    out: list[str] = []
    seen: set[str] = set()
    for m in matches:
        # findall may return tuples when there are groups
        if isinstance(m, tuple):
            for part in m:
                if part and part not in seen:
                    seen.add(part)
                    out.append(part.strip())
        elif isinstance(m, str):
            if m and m not in seen:
                seen.add(m)
                out.append(m.strip())
        if len(out) >= 5:
            break
    return out


# ---------------------------------------------------------------------------
# YAML keyword append
# ---------------------------------------------------------------------------

KEYWORD_BLOCK_MARK = "# auto-improve symptom keywords (Stop-hook learned):"


def append_keywords_to_yaml(yaml_path: Path, phrases: list[str]) -> int:
    """Append unique phrases to the auto-improve keyword block inside the YAML
    frontmatter. Idempotent — returns the number of NEW phrases added.

    The block is created on first call. On subsequent calls existing phrases
    are deduplicated so the same prompt does not bloat the file.
    """
    if not phrases:
        return 0
    if not yaml_path.exists():
        return 0
    try:
        text = yaml_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    parts = text.split("---", 2)
    if len(parts) < 3:
        return 0
    head, frontmatter, body = parts[0], parts[1], parts[2]

    # Locate the keyword block inside the frontmatter (or create it).
    existing: set[str] = set()
    block_lines: list[str] = []
    block_present = KEYWORD_BLOCK_MARK in frontmatter
    if block_present:
        # Walk the frontmatter line-by-line to extract the existing block.
        in_block = False
        new_front: list[str] = []
        for line in frontmatter.splitlines():
            if line.strip().startswith(KEYWORD_BLOCK_MARK):
                in_block = True
                block_lines.append(line)
                continue
            if in_block:
                # Block continues while lines are `#` comments starting with `#   `
                if line.startswith("#   "):
                    phrase = line[4:].strip()
                    if phrase:
                        existing.add(phrase)
                    block_lines.append(line)
                    continue
                else:
                    in_block = False
            new_front.append(line)
        # Re-insert the block at the end of frontmatter (before trailing newline).
        # We will rebuild below.
        frontmatter = "\n".join(new_front)

    added = [p for p in phrases if p and p not in existing]
    if not added:
        return 0

    # Build updated block.
    block_out = [KEYWORD_BLOCK_MARK]
    for p in sorted(existing | set(added)):
        block_out.append(f"#   {p}")

    new_frontmatter = frontmatter.rstrip() + "\n" + "\n".join(block_out) + "\n"
    new_text = head + "---" + new_frontmatter + "---" + body
    try:
        yaml_path.write_text(new_text, encoding="utf-8")
    except OSError:
        return 0
    return len(added)


# ---------------------------------------------------------------------------
# Re-ingest
# ---------------------------------------------------------------------------


def reingest() -> bool:
    """Re-invoke agent_kms.ingest as a subprocess. Returns True on success."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "agent_kms.ingest"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            _log(f"reingest failed rc={proc.returncode}: {proc.stderr[-500:]}")
            return False
        return True
    except Exception as exc:
        _log(f"reingest exception: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True, type=Path)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="Detect gaps and log them, but do not edit YAMLs or reingest.")
    args = ap.parse_args()

    _log(f"start session={args.session_id} transcript={args.transcript}")

    prompts = parse_user_messages(args.transcript)
    if not prompts:
        _log("no user messages parsed; exit")
        return 0
    _log(f"parsed {len(prompts)} user message(s)")

    # Aggregate gaps across prompts (de-dupe by rule_id × prompt index).
    gaps: list[dict] = []
    for prompt in prompts:
        if len(gaps) >= MAX_GAPS_PER_SESSION:
            break
        # Cheap pre-filter: only run retrieve if a rule regex matches the
        # prompt. retrieve() is the expensive call (model load + Qdrant).
        candidate_rules = [r for r in load_gap_rules() if r["regex"].search(prompt)]
        if not candidate_rules:
            continue
        top_sources = run_retrieve(prompt)
        if not top_sources:
            continue
        for rule in candidate_rules:
            if rule["expected_source"] not in top_sources:
                phrases = extract_symptom_phrases(prompt, rule["regex"])
                gaps.append({
                    "rule_id": rule["id"],
                    "expected_source": rule["expected_source"],
                    "prompt_excerpt": prompt[:200],
                    "phrases": phrases,
                })

    if not gaps:
        _log("no retrieval gaps detected; exit")
        return 0
    _log(f"detected {len(gaps)} gap(s)")

    if args.dry_run:
        for g in gaps:
            _log(f"  dry-run gap rule={g['rule_id']} expected={g['expected_source']} "
                 f"phrases={g['phrases']}")
        return 0

    # Apply YAML edits.
    total_added = 0
    touched_files: set[Path] = set()
    for g in gaps:
        yaml_path = REPO_ROOT / g["expected_source"]
        added = append_keywords_to_yaml(yaml_path, g["phrases"])
        if added > 0:
            total_added += added
            touched_files.add(yaml_path)
            _log(f"  appended {added} phrase(s) to {yaml_path.name}: {g['phrases']}")
        else:
            _log(f"  no new phrases for {yaml_path.name} (all already present)")

    if total_added == 0:
        _log("no new phrases written; skip reingest")
        return 0

    _log(f"reingesting after {total_added} phrase append(s) across {len(touched_files)} file(s)")
    ok = reingest()
    _log(f"reingest {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
