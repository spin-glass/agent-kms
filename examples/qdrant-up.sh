#!/usr/bin/env bash
# Start a persistent local Qdrant container for agent-kms.
set -e

NAME="${AGENT_KMS_QDRANT_NAME:-agent-kms-qdrant}"
DATA_DIR="${AGENT_KMS_QDRANT_DATA:-$HOME/qdrant_data}"
PORT="${AGENT_KMS_QDRANT_PORT:-6333}"

mkdir -p "$DATA_DIR"

if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}\$"; then
  echo "container ${NAME} exists; starting if stopped"
  docker start "${NAME}" >/dev/null
else
  docker run -d \
    --name "${NAME}" \
    --restart unless-stopped \
    -p "${PORT}:6333" \
    -v "${DATA_DIR}:/qdrant/storage" \
    qdrant/qdrant:latest >/dev/null
  echo "started container ${NAME}"
fi

echo "Qdrant URL: http://localhost:${PORT}"
curl -sSf "http://localhost:${PORT}/collections" || {
  echo "warning: Qdrant did not respond within timeout"
  exit 1
}
