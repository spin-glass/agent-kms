"""Multi-provider LLM abstraction for the RAG pipeline.

Default chain: Gemini 2.5 Flash (primary, free-tier 250 RPD) -> Haiku 4.5
fallback. Configurable via env:

    RAG_PROVIDER          auto | gemini | haiku   (default auto)
    RAG_PROVIDER_FALLBACK gemini | haiku | none   (default haiku)
    GEMINI_API_KEY        AIza...
    GEMINI_MODEL          gemini-2.5-flash         (default)
    ANTHROPIC_API_KEY     sk-ant-...
    ANTHROPIC_MODEL       claude-haiku-4-5-20251001 (default)

`generate(prompt, ...)` returns `LLMResult` with the text and provenance,
trying providers in order and falling back on rate-limit / auth / network
errors.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

_PLACEHOLDER_PATTERNS = (
    "REPLACE_ME",
    "AIzaSy...",
    "sk-ant-...",
    "your-key-here",
)


def _is_placeholder(key: str) -> bool:
    if not key:
        return True
    return any(pat in key for pat in _PLACEHOLDER_PATTERNS)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str


class _GeminiProvider:
    name = "gemini"

    def __init__(self, model: str | None = None):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if _is_placeholder(api_key):
            raise RuntimeError("GEMINI_API_KEY missing or placeholder")
        # Lazy import so absent SDK only breaks Gemini path, not Haiku
        from google import genai

        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self._genai = genai
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
        response_schema: object | None = None,
    ) -> LLMResult:
        from google.genai import types

        cfg_kwargs: dict = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            # Disable thinking by default. Gemini 2.5 Flash spends "thinking
            # tokens" out of max_output_tokens before emitting visible output;
            # with a tight budget the visible text comes back empty. Lesson
            # extraction / refine / D2Q are structured tasks that do not
            # benefit from extended reasoning.
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
        }
        if json_mode or response_schema is not None:
            cfg_kwargs["response_mime_type"] = "application/json"
        if response_schema is not None:
            cfg_kwargs["response_schema"] = response_schema
        config = types.GenerateContentConfig(**cfg_kwargs)
        response = self.client.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        text = response.text or ""
        return LLMResult(text=text, provider=self.name, model=self.model)


class _AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str | None = None):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if _is_placeholder(api_key):
            raise RuntimeError("ANTHROPIC_API_KEY missing or placeholder")
        import anthropic

        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_mode: bool = False,
        response_schema: object | None = None,
    ) -> LLMResult:
        # Anthropic has no native JSON mode flag; rely on prompt instruction.
        # response_schema is silently ignored on this provider (Gemini-only).
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        return LLMResult(text=text, provider=self.name, model=self.model)


def _construct(name: str):
    if name == "gemini":
        return _GeminiProvider()
    if name in ("haiku", "anthropic", "claude"):
        return _AnthropicProvider()
    raise RuntimeError(f"unknown provider: {name}")


def _build_chain() -> list:
    primary = os.environ.get("RAG_PROVIDER", "auto").lower()
    fallback = os.environ.get("RAG_PROVIDER_FALLBACK", "haiku").lower()

    chain: list = []
    if primary == "auto":
        for name in ("gemini", "haiku"):
            try:
                chain.append(_construct(name))
            except Exception as e:
                logger.info("provider %s unavailable: %s", name, e)
        return chain

    try:
        chain.append(_construct(primary))
    except Exception as e:
        logger.warning("primary provider %s unavailable: %s", primary, e)

    if fallback != "none" and fallback != primary:
        try:
            chain.append(_construct(fallback))
        except Exception as e:
            logger.info("fallback provider %s unavailable: %s", fallback, e)

    return chain


def generate(
    prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    json_mode: bool = False,
    response_schema: object | None = None,
    gemini_model: str | None = None,
    anthropic_model: str | None = None,
) -> LLMResult:
    """Per-call tier override:
        gemini_model / anthropic_model — pin the model for this call only.
        response_schema — Gemini-only structured-output schema (Pydantic /
            list[str] / dict). Ignored by Anthropic; rely on prompt for JSON.
    """
    chain = _build_chain()
    if not chain:
        raise RuntimeError(
            "No LLM provider available. Set GEMINI_API_KEY (free tier ok) "
            "and/or ANTHROPIC_API_KEY in tools/rag/.env"
        )

    # Apply per-call model overrides
    for p in chain:
        if p.name == "gemini" and gemini_model:
            p.model = gemini_model
        elif p.name == "anthropic" and anthropic_model:
            p.model = anthropic_model

    last_err: Exception | None = None
    for provider in chain:
        try:
            return provider.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
                response_schema=response_schema,
            )
        except Exception as e:
            logger.warning(
                "provider %s/%s failed (%s: %s); trying next",
                provider.name,
                provider.model,
                type(e).__name__,
                e,
            )
            last_err = e

    raise RuntimeError(f"all LLM providers failed; last error: {last_err}")


def is_available() -> bool:
    return bool(_build_chain())


def chain_summary() -> str:
    chain = _build_chain()
    if not chain:
        return "(no provider available)"
    return " -> ".join(f"{p.name}:{p.model}" for p in chain)
