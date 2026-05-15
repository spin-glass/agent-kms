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

TRANSCRIPT=$(printf '%s' "$INPUT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('transcript_path', ''), end='')
except Exception:
    pass
" 2>/dev/null)

STOP_ACTIVE=$(printf '%s' "$INPUT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('stop_hook_active', ''), end='')
except Exception:
    pass
" 2>/dev/null)

[ -z "$TRANSCRIPT" ] && exit 0
[ ! -f "$TRANSCRIPT" ] && exit 0
[ "$STOP_ACTIVE" = "True" ] && exit 0

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

# Detach so the 5s Stop-hook timeout doesn't kill the long-running work
if command -v setsid >/dev/null 2>&1; then
  setsid nohup agent-kms extract-lessons --transcript "$TRANSCRIPT" \
    >>"$LOG_FILE" 2>&1 < /dev/null &
else
  nohup agent-kms extract-lessons --transcript "$TRANSCRIPT" \
    >>"$LOG_FILE" 2>&1 < /dev/null &
fi
disown 2>/dev/null
exit 0
