#!/usr/bin/env bash
# =============================================================================
# Claude Code hook: UserPromptSubmit — agent-kms auto retrieve
# =============================================================================
#
# 1. Read the user prompt from stdin (Claude Code hook protocol).
# 2. Apply a cheap keyword gate (no LLM, no network) — only fire on action-
#    oriented prompts. Greeting / chat-only prompts skip silently.
# 3. Call ``agent-kms retrieve --json`` for threshold-passing chunks.
# 4. Emit a human-readable ``<agent-kms-context>...</agent-kms-context>``
#    block on stdout — Claude Code prepends this to the next assistant turn.
# 5. Append a structured JSONL record to
#    ``~/.claude/logs/agent-kms-retrieve.jsonl`` so the Stop-hook
#    effectiveness pass can later tell which surfaced chunks were used.
#
# Skips silently (exit 0) on:
#   - empty prompt
#   - Qdrant unreachable
#   - prompt missing fix/implementation/verify intent keywords
#
# Configure threshold via AGENT_KMS_RETRIEVE_THRESHOLD (default 0.83).
# =============================================================================

set +e

LOG_DIR="$HOME/.claude/logs"
mkdir -p "$LOG_DIR"

INPUT=$(cat)
[ -z "$INPUT" ] && exit 0

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
curl -sSf -m 1 "${QDRANT_URL%/}/collections" >/dev/null 2>&1 || exit 0

THRESHOLD="${AGENT_KMS_RETRIEVE_THRESHOLD:-0.83}"

# Single Python pass: parse hook input, keyword-gate, call agent-kms,
# emit context block, append JSONL log. Keeping it in one subshell avoids
# fragile bash escaping around prompts that contain quotes / newlines.
#
# NOTE on plumbing: pass $INPUT via the HOOK_INPUT env var rather than
# piping to stdin. `python3 - <<'PY'` already uses stdin for the script
# body (heredoc) — any earlier `printf | python3` is silently overridden
# by bash's redirection precedence, and json.load(sys.stdin) then reads
# EOF and the hook becomes a permanent no-op.
HOOK_INPUT="$INPUT" THRESHOLD="$THRESHOLD" python3 - <<'PY'
import json, os, re, subprocess, sys, time

THRESHOLD = os.environ.get("THRESHOLD", "0.83")
LOG_PATH = os.path.expanduser("~/.claude/logs/agent-kms-retrieve.jsonl")

try:
    data = json.loads(os.environ.get("HOOK_INPUT", ""))
except Exception:
    sys.exit(0)

prompt = (data.get("prompt") or "").strip()
session_id = data.get("session_id") or ""
if not prompt:
    sys.exit(0)

# Action-intent keyword gate (Japanese + English). Same vocabulary the
# original auto-rag-retrieve used so behaviour is unchanged when migrating.
KW_RE = re.compile(
    r"修正|治|直|壊|バグ|不具合|エラー|問題|改善|"
    r"fix|bug|broken|error|issue|problem|"
    r"実装|作る|作成|追加|新規|移植|編集|変更|"
    r"implement|create|add|new|port|modify|edit|write|"
    r"検証|確認|調査|分析|比較|デバッグ|リファクタ|"
    r"refactor|debug|verify|check|analyze|compare|review|inspect",
    re.IGNORECASE,
)
if not KW_RE.search(prompt):
    sys.exit(0)

# Single retrieve call, JSON output. We re-format for the context block
# below — avoids a double model-load round trip.
try:
    res = subprocess.run(
        ["agent-kms", "retrieve", "--json", prompt, "--threshold", THRESHOLD],
        capture_output=True, text=True, timeout=20,
    )
except Exception:
    sys.exit(0)

if res.returncode != 0 or not res.stdout.strip():
    sys.exit(0)

try:
    hits = json.loads(res.stdout)
except json.JSONDecodeError:
    sys.exit(0)

if not hits:
    sys.exit(0)

# Context block for Claude.
print("<agent-kms-context>")
print("Top relevant knowledge for this prompt (auto-retrieved):")
for r in hits:
    src = r.get("source", "?")
    heading = r.get("heading", "")
    score = float(r.get("score", 0.0))
    sev = r.get("severity", "")
    app = r.get("applicability", "")
    text = " ".join((r.get("text") or "").split())[:240]
    print(
        f"  • [{r.get('source_type', '?')} {sev}/{app}] "
        f"{src} #{heading} (score={score:.3f})"
    )
    print(f"    {text}")
    print()
print("</agent-kms-context>")

# Append the structured event for later effectiveness analysis. Keep only
# the fields needed downstream (no full chunk body — that bloats the log).
try:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session_id": session_id,
            "query": prompt[:500],
            "threshold": float(THRESHOLD),
            "hits": [{
                "source": r.get("source", ""),
                "heading": r.get("heading", ""),
                "source_type": r.get("source_type", ""),
                "score": float(r.get("score", 0.0)),
            } for r in hits],
        }, ensure_ascii=False) + "\n")
except OSError:
    # Logging failure is non-fatal — the context block was already printed.
    pass
PY
exit 0
