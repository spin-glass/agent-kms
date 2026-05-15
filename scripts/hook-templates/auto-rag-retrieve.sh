#!/usr/bin/env bash
# =============================================================================
# Claude Code hook: UserPromptSubmit — agent-kms auto retrieve
# =============================================================================
#
# Reads the user prompt from stdin (Claude Code hook protocol), runs a
# relevance retrieve via ``agent-kms retrieve``, and emits the formatted
# results on stdout. Claude Code prepends UserPromptSubmit stdout to the
# next assistant turn, so this surfaces relevant past knowledge without
# the model having to invoke a tool.
#
# Skips silently (exit 0) on:
#   - empty prompt
#   - Qdrant unreachable
#   - prompt missing fix/implementation/verify intent keywords
#
# Configure threshold via AGENT_KMS_RETRIEVE_THRESHOLD (default 0.93).
# Logs to $HOME/.claude/logs/agent-kms-auto-retrieve.log
# =============================================================================

set +e

LOG_DIR="$HOME/.claude/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/agent-kms-auto-retrieve.log"

INPUT=$(cat)
[ -z "$INPUT" ] && exit 0

USER_PROMPT=$(printf '%s' "$INPUT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('prompt', ''), end='')
except Exception:
    pass
" 2>/dev/null)

[ -z "$USER_PROMPT" ] && exit 0

# Skip if Qdrant unreachable
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
curl -sSf -m 1 "${QDRANT_URL%/}/collections" >/dev/null 2>&1 || exit 0

# Keyword gate — only fire on action-oriented prompts
echo "$USER_PROMPT" | grep -qiE \
  '修正|治|直|壊|バグ|不具合|エラー|問題|改善|fix|bug|broken|error|issue|problem|実装|作る|作成|追加|新規|移植|編集|変更|implement|create|add|new|port|modify|edit|write|検証|確認|調査|分析|比較|デバッグ|リファクタ|refactor|debug|verify|check|analyze|compare|review|inspect' \
  || exit 0

THRESHOLD="${AGENT_KMS_RETRIEVE_THRESHOLD:-0.93}"

OUTPUT=$(agent-kms retrieve "$USER_PROMPT" --threshold "$THRESHOLD" 2>>"$LOG_FILE")
RC=$?
[ $RC -ne 0 ] && exit 0
[ -z "$OUTPUT" ] && exit 0
[ "$OUTPUT" = "(no chunks above threshold)" ] && exit 0

echo "<agent-kms-context>"
echo "Top relevant knowledge for this prompt (auto-retrieved):"
echo "$OUTPUT"
echo "</agent-kms-context>"
exit 0
