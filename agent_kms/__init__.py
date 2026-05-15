"""agent-kms: Generic AI agent knowledge management system.

Vector-backed (Qdrant) retrieval + Stop-hook lesson extraction designed to
loop session-level insights back into future Claude Code / agent sessions.

Auto-loads ``$AGENT_KMS_ENV_FILE`` (default: ``./.env`` in current working
directory) at package import so any module gets ``GEMINI_API_KEY`` /
``ANTHROPIC_API_KEY`` / ``QDRANT_URL`` / ``AGENT_KMS_KNOWLEDGE_ROOT``
without the caller having to source the file.

``override=False`` keeps shell env winning, so user-level secrets are not
silently shadowed by a placeholder .env.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_DOTENV_PATH = Path(
    os.environ.get("AGENT_KMS_ENV_FILE", str(Path.cwd() / ".env"))
).expanduser()
if _DOTENV_PATH.exists():
    load_dotenv(_DOTENV_PATH, override=False)

__version__ = "0.1.0"
