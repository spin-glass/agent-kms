"""LLM-based refine: filter and distill retrieved chunks for a planning query.

Takes raw retrieve output (top-N chunks) and produces:
- per-chunk applicability decision (apply / skip + reason)
- consolidated summary that AI consumers paste into planning context

This is the "consumer-facing context" layer between raw retrieval and the
AI planner that calls `cocos-port`. A single LLM call covers both steps.

Output schema:
{
  "summary": "...",                      # 200-800 字の domain-independent migration principles 集約
  "applicable": [
    {"source": "...", "reason": "..."},  # 採用 chunk + 採用理由
  ],
  "skipped": [
    {"source": "...", "reason": "..."},  # 棄却 chunk + 棄却理由
  ]
}
"""

from __future__ import annotations

import json
import os
import re

from .llm import generate as llm_generate

REFINE_PROMPT = """以下は Cocos2d-x → Unity 移植 RAG システムが query に対して取得した top chunks です。
あなたのタスクは、これらの chunk を以下 2 つの観点で評価し、AI 計画者に提供する distilled context を生成することです。

## 入力
query: {query}

candidate chunks (各 chunk は migration 知見、instinct、or session lesson):
{chunks_json}

## あなたが行うこと
1. 各 chunk を query に対して "applicable / skip" 判定
   - applicable: 移植計画判断に直接 influence する domain-independent migration principles または specific gotchas
   - skip: query と無関係 (false positive) または lexical noise
2. applicable な chunk から **本 query に対する移植時に AI が思い出すべき要点** を 200-800 字で集約
   - vague 表現禁止 (例: "気をつける" は NG)
   - 具体的 actionable principles のみ抽出
   - source citation を [filename] 形式で添える
3. skipped chunks には簡潔な reason を付ける

## 出力 (JSON のみ)
{{
  "summary": "...",
  "applicable": [{{"source": "<filename>", "reason": "..."}}, ...],
  "skipped": [{{"source": "<filename>", "reason": "..."}}, ...]
}}
"""


def refine(query: str, chunks: list[dict], max_tokens: int = 2048) -> dict:
    """Distill retrieved chunks into AI-consumer-ready context.

    `chunks` is the list returned by `retrieve()`; expects keys
    `text`, `source`, `source_type`, `score`.
    Returns a dict with keys `summary`, `applicable`, `skipped`, `provider`,
    `model`. On parse failure returns a degraded dict with all chunks marked
    applicable and the raw text as summary.
    """
    if not chunks:
        return {
            "summary": "(no candidates retrieved)",
            "applicable": [],
            "skipped": [],
            "provider": "skipped",
            "model": "n/a",
        }

    # Limit per-chunk text to keep prompt size reasonable
    compact = [
        {
            "source": c["source"].split("/")[-1],
            "source_type": c.get("source_type", ""),
            "text": c["text"][:1200],
        }
        for c in chunks
    ]
    chunks_json = json.dumps(compact, ensure_ascii=False, indent=2)

    # Refine = balanced tier (Flash). Quality matters more here than fast volume.
    balanced_model = os.environ.get("GEMINI_MODEL_BALANCED")
    result = llm_generate(
        REFINE_PROMPT.format(query=query, chunks_json=chunks_json),
        max_tokens=max_tokens,
        temperature=0.0,
        json_mode=True,
        gemini_model=balanced_model,
    )

    text = (result.text or "").strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {
            "summary": text[:1500],
            "applicable": [{"source": c["source"], "reason": "parse failure, kept raw"} for c in compact],
            "skipped": [],
            "provider": result.provider,
            "model": result.model,
            "parse_error": True,
        }

    return {
        "summary": parsed.get("summary", ""),
        "applicable": parsed.get("applicable", []),
        "skipped": parsed.get("skipped", []),
        "provider": result.provider,
        "model": result.model,
    }
