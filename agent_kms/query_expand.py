"""Query expansion for retrieve queries.

Two strategies:
  - ``template_expand`` — append a fixed list of "universal aspects" loaded
    from the active preset's ``prompts/query_expand_template.md``. Zero LLM
    cost; lifts retrieve hits for aspect-related chunks.
  - ``llm_expand`` — ask an LLM (Gemini Flash-Lite by default) to enumerate
    the universal aspects implied by the user's planning query.
"""

from __future__ import annotations

import os
import re

from .config import load_prompt
from .llm import generate as llm_generate

_TEMPLATE_NAME = "query_expand_template"
_LLM_PROMPT_NAME = "query_expand_llm"


def _load_template() -> str:
    try:
        return load_prompt(_TEMPLATE_NAME).strip()
    except FileNotFoundError:
        return ""


def template_expand(query: str) -> str:
    """Static template expansion. Use when LLM cost / latency is critical."""
    template = _load_template()
    if not template:
        return query
    return f"{query}. Related universal aspects: {template}"


def llm_expand(query: str) -> str:
    """LLM-based dynamic expansion. Falls back to ``template_expand`` on failure."""
    fast_model = os.environ.get("GEMINI_MODEL_FAST")
    try:
        prompt = load_prompt(_LLM_PROMPT_NAME).format(query=query)
    except FileNotFoundError:
        return template_expand(query)
    try:
        result = llm_generate(
            prompt,
            max_tokens=512,
            temperature=0.0,
            json_mode=False,
            gemini_model=fast_model,
        )
        text = (result.text or "").strip()
        text = re.sub(r"^[\"'「『]?\s*", "", text)
        text = re.sub(r"\s*[\"'」』]?$", "", text)
        if text and len(text) > len(query):
            return text
    except Exception:
        pass
    return template_expand(query)
