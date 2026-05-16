"""FastMCP server exposing ``retrieve_for_planning`` to Claude Code / agents.

Run via the ``agent-kms serve`` CLI (or directly: ``python -m agent_kms.server``).

Register in Claude Code MCP config (``~/.claude.json`` or workspace settings):

    {
      "mcpServers": {
        "agent-kms": {
          "command": "agent-kms",
          "args": ["serve"]
        }
      }
    }
"""

from __future__ import annotations

import os

from fastmcp import FastMCP

from .retrieve import retrieve

MCP_NAME = os.environ.get("AGENT_KMS_MCP_NAME", "agent-kms")
mcp = FastMCP(MCP_NAME)


@mcp.tool()
def retrieve_for_planning(query: str, score_threshold: float = 0.83) -> list[dict]:
    """Retrieve relevant knowledge chunks above a relevance threshold.

    Used by agents (Claude Code, etc.) at the start of planning steps to
    surface past lessons / known issues / instincts that match the query.

    Args:
        query: Natural-language description of the task or symptom.
        score_threshold: Boosted-cosine floor. 0.83 is tuned for the
            current default embedding model (``cl-nagoya/ruri-v3-310m``)
            on Japanese-heavy corpora; lower (~0.78) to broaden, raise
            (~0.90) for strict matches. Calibrate per corpus with
            ``agent-kms calibrate-threshold``.
            **No count cap** — every chunk above the threshold is returned.

    Returns:
        List of dicts with keys: ``text``, ``source``, ``source_type``,
        ``severity``, ``applicability``, ``heading``, ``score``.
        Sorted descending by boosted score.
    """
    return retrieve(query, score_threshold=score_threshold)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
