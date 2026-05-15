"""D2Q (Document-to-Query) synthetic query generator.

For each ingest chunk, generate N synthetic queries that the chunk would
answer. Embedding the synthetic queries (instead of the chunk body) gives
Q-Q symmetric retrieval at planning time -- a real user query matches
synthetic Qs in the same `query: ...` embedding submanifold.

Idempotent via on-disk cache: chunk content hash -> [Q1..QN]. Re-running
ingest does not re-call the LLM unless chunk text changes.

Plan α (cost-optimised): use Gemini Flash-Lite with Few-shot examples and
strict response_schema (list[str]). Falls back to Haiku via the chain when
Gemini hits rate-limit / parse failure / network error.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from .llm import generate as llm_generate

CACHE_PATH = Path(__file__).resolve().parent.parent / ".synthetic_queries.json"

PROMPT = """以下は Cocos2d-x → Unity 移植プロジェクトのナレッジ chunk です。
**この chunk の本文に specific に答えられる query** を {n} 件生成してください。

## 厳守
- query は **chunk 内で実際に言及されている** 技術詳細 (関数名 / クラス名 / ファイル名 / 数値 / エラー名 / 失敗 pattern 名 / 具体手法 / Linear ticket 番号) を 1 つ以上含む
- 「Cocos2d-x から Unity への UI 移植で〜」「ベストプラクティスは?」のような **どの chunk にも当てはまる generic query は禁止** (D2Q 索引として無効になる)
- chunk が言及していない事項を query に含めない (架空話 NG)
- 自然言語の質問形式、15-100 字 (日本語または英語、chunk の主言語に合わせる)
- forbidden vocab (skeleton/placeholder/dummy/scope-narrow) は使わない

## Few-shot 例 (3 件)

### 例 1: DOTween chunk
chunk: "DOTween で新しいアニメーションを開始する前に、対象 rt の既存 tween を rt.DOKill() で cancel する。
DOKill しないと前の tween は完走し、特に OnComplete が SetActive(false) を呼ぶケースで状態破壊が起きる。"
良 Q: ["DOTween で rt.DOKill() を呼ばないと OnComplete callback はどう発火する?",
       "新 tween 開始前に既存 tween を kill しないとどんな状態破壊が起きる?",
       "OnComplete で SetActive(false) を呼ぶ tween パターンの安全な書き方は?",
       "rt.DOKill() を忘れた場合の典型的な race condition は?",
       "DOTween で前の tween が完走して状態破壊する具体例は?"]

### 例 2: ScrollRect chunk
chunk: "ScrollRect + VerticalLayoutGroup + ContentSizeFitter の child には LayoutElement 必須。
preferredHeight を指定しないと行高さが 0 に潰れて全行重なり表示になる。"
良 Q: ["ScrollRect の child に LayoutElement を付け忘れると preferredHeight はどう計算される?",
       "VerticalLayoutGroup + ContentSizeFitter で行が重なる原因と対処法は?",
       "preferredHeight 指定なしで row が 0 に潰れるのを回避する方法は?",
       "ScrollRect の virtualized list で LayoutElement 必須な理由は?",
       "全行重なり表示の bug を引き起こす ScrollRect 設定は?"]

### 例 3: setLabel chunk
chunk: "Cocos の `setLabel(x, y)` は y=CENTER だが Unity の RectTransform.anchoredPosition は TOP-LEFT 起点。
CreateCocosLabel helper が h/2 オフセットを自動付与して anchor 差を吸収する。"
良 Q: ["setLabel(x, y) と RectTransform.anchoredPosition の anchor 起点はどう違う?",
       "CreateCocosLabel helper が h/2 オフセットを自動付与する理由は?",
       "Cocos の y=CENTER と Unity の TOP-LEFT 差を helper なしで実装すると label がどう h/2 ずれる?",
       "setLabel の y 引数を Unity に直接そのまま渡すとどんな視覚不一致が起きる?",
       "anchor 規約差を吸収する CreateCocosLabel の正しい使い方は?"]

## 悪い例 (chunk specific でなく generic)
- "Unity でアニメーションを実装するには?"  ← chunk の specific term ゼロ
- "Cocos→Unity の落とし穴は?"            ← どの chunk にも当てはまる
- "ScrollRect の使い方は?"                 ← chunk が触れていない一般質問

## 対象 chunk
{chunk_text}

JSON 配列のみ出力 (文字列 {n} 件):"""

# Strict schema for Gemini structured output. List of strings.
RESPONSE_SCHEMA = list[str]


def _load_cache() -> dict[str, list[str]]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, list[str]]) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def generate_qs_for_chunk(chunk_text: str, n: int = 5) -> list[str]:
    """Single LLM call -> n synthetic queries for one chunk.

    Uses GEMINI_MODEL_FAST (Flash-Lite by default, Plan α) with Few-shot
    examples and strict list[str] response schema. Falls back to Haiku via
    the chain when Gemini fails.
    """
    fast_model = os.environ.get("GEMINI_MODEL_FAST")
    result = llm_generate(
        PROMPT.format(n=n, chunk_text=chunk_text[:3000]),
        max_tokens=1024,
        temperature=0.0,
        json_mode=True,
        response_schema=RESPONSE_SCHEMA,
        gemini_model=fast_model,
    )
    text = (result.text or "").strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    try:
        qs = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(qs, list):
        return []
    # Accept both shapes:
    #   ["query1", "query2", ...]                     -- desired
    #   [{"query": "...", "reason": "..."}, ...]     -- some models prefer
    out: list[str] = []
    for q in qs:
        if isinstance(q, str) and q.strip():
            out.append(q.strip())
        elif isinstance(q, dict):
            for key in ("query", "q", "text", "question"):
                v = q.get(key)
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
                    break
    return out[:n]


def generate_qs_batch(
    chunks: list[dict],
    n_per_chunk: int = 5,
    max_workers: int = 8,
) -> dict[str, list[str]]:
    """Generate Qs for all chunks, using on-disk cache + thread pool.

    Each chunk -> 1 LLM call (which produces N Qs). Calls are I/O-bound HTTP
    requests, so a thread pool of 8 workers parallelises ~8x without GIL pain.
    Cache writes are serialised through a lock.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache = _load_cache()
    cache_lock = threading.Lock()
    out: dict[str, list[str]] = {}

    todo: list[tuple[str, str]] = []  # (chunk_hash, chunk_text)
    cache_hits = 0
    for c in chunks:
        h = chunk_hash(c["text"])
        if h in cache and len(cache[h]) >= n_per_chunk:
            out[h] = cache[h][:n_per_chunk]
            cache_hits += 1
        else:
            todo.append((h, c["text"]))

    print(f"synthetic_q: cache_hits={cache_hits}, todo={len(todo)}, workers={max_workers}")

    completed = 0

    def _worker(item: tuple[str, str]) -> tuple[str, list[str]]:
        h, text = item
        return h, generate_qs_for_chunk(text, n=n_per_chunk)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker, item) for item in todo]
        for fut in as_completed(futures):
            try:
                h, qs = fut.result()
            except Exception as e:
                print(f"  worker failed: {type(e).__name__}: {e}")
                continue
            with cache_lock:
                if qs:
                    cache[h] = qs
                    out[h] = qs
                else:
                    out[h] = []
                completed += 1
                if completed % 25 == 0 or completed == len(todo):
                    _save_cache(cache)
                    print(f"  cached {completed}/{len(todo)} new (total cache {len(cache)})")

    with cache_lock:
        _save_cache(cache)
    print(f"synthetic_q: total cached={len(cache)}, processed_now={completed}")
    return out
