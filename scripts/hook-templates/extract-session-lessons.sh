#!/usr/bin/env bash
# =============================================================================
# Claude Code hook: Stop — agent-kms session lesson extractor
# =============================================================================
#
# Spawns ``agent-kms extract-lessons`` as a detached background process so
# the user's main session is not blocked by the 5s hook timeout. The heavy
# work (embedding model load + LLM call + upsert) runs out of band.
#
# Skips silently on:
#   - transcript missing
#   - stop_hook_active true (recursion guard)
#   - Qdrant unreachable
#   - no LLM provider configured
#       (no GEMINI / ANTHROPIC key AND RAG_PROVIDER is not ``ollama``)
#
# Logs to $HOME/.claude/logs/agent-kms-stop.log
# =============================================================================

set +e

LOG_DIR="$HOME/.claude/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/agent-kms-stop.log"

INPUT=$(cat)
[ -z "$INPUT" ] && exit 0

PARSED=$(printf '%s' "$INPUT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('transcript_path', ''))
    print(d.get('stop_hook_active', ''))
    print(d.get('session_id', ''))
except Exception:
    pass
" 2>/dev/null)
TRANSCRIPT=$(printf '%s\n' "$PARSED" | sed -n '1p')
STOP_ACTIVE=$(printf '%s\n' "$PARSED" | sed -n '2p')
SESSION_ID=$(printf '%s\n' "$PARSED" | sed -n '3p')

[ -z "$TRANSCRIPT" ] && exit 0
[ ! -f "$TRANSCRIPT" ] && exit 0
[ "$STOP_ACTIVE" = "True" ] && exit 0
# Fallback for older Claude Code versions that did not pass session_id:
# derive it from the transcript filename stem (Claude Code uses the UUID
# as the JSONL basename), so effectiveness lookup still works.
if [ -z "$SESSION_ID" ]; then
  SESSION_ID=$(basename "$TRANSCRIPT" .jsonl)
fi

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
curl -sSf -m 1 "${QDRANT_URL%/}/collections" >/dev/null 2>&1 || exit 0

# LLM gate: skip only when no provider is configured at all.
# - cloud: at least one of GEMINI_API_KEY / ANTHROPIC_API_KEY non-empty
# - local: RAG_PROVIDER == "ollama" (Ollama needs no key, just a running daemon)
if [ "$RAG_PROVIDER" != "ollama" ] \
   && [ -z "$GEMINI_API_KEY" ] \
   && [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "[$(date +%FT%T)] no LLM provider configured, skipping" >>"$LOG_FILE"
  exit 0
fi

# Detach so the 5s Stop-hook timeout doesn't kill the long-running work.
# All three passes run sequentially inside one detached shell so they share
# the LLM + embedding model load cost:
#   1. extract-lessons   — mine new lessons from the transcript tail
#   2. effectiveness     — summarise USED / UNUSED retrieved chunks
#   3. improve           — detect retrieval gaps & widen instinct keywords
SPAWN_PREFIX=""
command -v setsid >/dev/null 2>&1 && SPAWN_PREFIX="setsid"
TRANSCRIPT="$TRANSCRIPT" SESSION_ID="$SESSION_ID" \
  $SPAWN_PREFIX nohup bash -c '
    agent-kms extract-lessons --transcript "$TRANSCRIPT"
    echo "---"
    agent-kms effectiveness --transcript "$TRANSCRIPT" --session-id "$SESSION_ID"
    echo "---"
    agent-kms improve --transcript "$TRANSCRIPT" --session-id "$SESSION_ID"
  ' >>"$LOG_FILE" 2>&1 < /dev/null &
disown 2>/dev/null
exit 0
