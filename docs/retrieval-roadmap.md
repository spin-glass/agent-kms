# Retrieval Roadmap

Planned improvements to agent-kms's retrieval pipeline beyond the
current `dense cosine + severity_boost + applicability_penalty +
threshold` flow. Each item is captured with:

- **the problem** it solves (so we know when we've earned the right to do it)
- **mechanism** (one-paragraph summary)
- **paper / source** (so future-us can re-read instead of re-thinking)
- **agent-kms implementation site** (where the code change goes)
- **validation plan** in [rag-evaluation-jp](https://github.com/spin-glass/rag-evaluation-jp)
  (no upstream change without empirical justification on a representative
  corpus — public benchmark or private validation per `EVAL_MODE`)

The current state — dense embedding (multilingual-e5-base or ruri-v3-310m,
768d) + boosts + a fixed score threshold — handles the easy case. The
items below address concrete failure modes that have been observed or
are likely to surface as the corpus grows.

---

## 0. Current state (baseline)

```
query text
  → template_expand (optional, keyword append)
  → encode_query (SentenceTransformer, 768d)
  → Qdrant cosine search, limit=200
  → + severity_boost  (critical +0.05, high +0.025)
  → + applicability_boost  (universal 0, conditional -0.005, topic-specific -0.015)
  → filter: boosted_score >= score_threshold (default 0.93, project usually 0.85)
  → return all passing
```

Known weaknesses, in rough priority order:

1. **Identifier / token queries** lose signal in dense embeddings (e.g.
   `@japan-ai/core-prisma/client`).
2. **Abstract documents** don't surface for specific queries (vocabulary
   mismatch — query says "売上が落ちた原因", doc says "収益性に影響").
3. **Cross-paraphrase recall** — same intent in different wording fails
   to match.
4. **No-evidence queries** still produce some chunks above threshold for
   topical-but-not-answering text.

Each roadmap item below targets one or more of these.

---

## 1. Hybrid retrieval (BM25 + Dense)

**Targets weaknesses (1) and (4).**

### Problem
Dense embeddings smear identifier tokens (filenames, function names,
literal flag strings) across the vocabulary. A query like
`@japan-ai/core-prisma/client` cannot fully recover the exact-match
signal a sparse retriever gives for free.

### Mechanism
Run dense + BM25 in parallel and fuse the rankings. Two fusion options:

- **Reciprocal Rank Fusion (RRF)** — `score = Σ 1/(k+rank_i)`. Simple,
  parameter-light, robust. Original paper:
  Cormack, Clarke, Büttcher — *Reciprocal Rank Fusion outperforms
  Condorcet and individual rank learning methods* (SIGIR 2009).
- **Linear combination** — `score = α·dense + β·bm25`. Needs tuning.

### State in agent-kms
`agent_kms/sparse.py` already wraps fastembed's `Qdrant/bm25` model
into the same `encode_passages` / `encode_query` shape as the dense
encoder. **Not currently called from anywhere** — pure scaffold.
Wiring up requires:

- `store.ensure_collection` — add `sparse_vectors_config` so the
  collection accepts both dense + sparse vectors.
- `ingest.upsert_chunks` — encode + upsert sparse vector alongside dense.
- `retrieve.retrieve` — switch to Qdrant's `Prefetch` API to run both
  queries and merge via RRF (server-side `FusionQuery` or client-side).

### Validation plan
New notebook `rag-evaluation-jp/notebooks/07_hybrid_retrieval.ipynb`:
sweep `dense-only / sparse-only / hybrid-RRF / hybrid-linear` on the
public corpus + the private corpus once it has enough identifier-heavy
queries (target N≥1000 chunks).

Anthropic's *Contextual Retrieval* writeup (2024-09) reported a 49%
reduction in retrieval failure when adding Contextual BM25 on top of
Contextual Embeddings — that's the headline number to beat / replicate.

### Cost
Ingest time ~1.5-2× (sparse encode is fast but still serial), retrieve
latency +50-100ms (parallelisable), Qdrant storage modest (sparse
vectors are sparse). Threshold needs re-calibration — the combined
score is no longer pure cosine.

---

## 2. Contextual Retrieval

**Targets weaknesses (1), (2), and (3).** Highest-impact single change
on abstract corpora per Anthropic's benchmark.

### Problem
A chunk like `市場環境の変化により収益性に影響が生じた` (from a section
midway through an annual-report document) is too abstract to match a
direct query like `売上が落ちた原因`. The chunk's surrounding context
(the company, the year, the topic) is invisible to retrieval.

### Mechanism
At ingest time, ask an LLM to write 2-3 sentences of context for each
chunk *given the whole document*, then prepend that context to the
chunk's embedded text. Source paper / writeup:

> Anthropic — *Introducing Contextual Retrieval* (2024-09).
> <https://www.anthropic.com/news/contextual-retrieval>

Reported failure-rate reductions: 35% (Contextual Embeddings alone),
49% (+ Contextual BM25), up to 67% (+ reranker).

Prompt template (translated to fit agent-kms's "match transcript
language" convention):

```
<document>
{whole_document}
</document>

Given the document above, write 2-3 sentences of context in the same
language as the document, anchoring this chunk's role / topic / scope
so it can be retrieved by queries that do not share the chunk's exact
vocabulary. Output the context only — no preamble, no header.

<chunk>
{chunk}
</chunk>
```

### State in agent-kms
Not started. Implementation site:

- New module `agent_kms/contextualize.py` — wraps `llm.generate` with
  the prompt above.
- `chunker.chunk_markdown_h2` — opt-in flag `contextualize=True` (per
  source). When enabled, the chunker passes the full document body
  alongside each chunk to the LLM and stores `context + chunk` as
  `embed_text` (the chunk body that ends up in the vector is the
  augmented version; the original body stays in the payload as `text`).
- `kms.toml` — per-source `contextualize = true` flag.

### Cost optimisation: prompt caching
The "whole document" stays the same across all chunks of one file.
With Anthropic's prompt cache (`cache_control: ephemeral`), only the
chunk-specific tail is re-billed → ~90% cost reduction for the per-file
batch. Gemini also supports context caching with similar ergonomics.

For local-Ollama setups (the current default in this project) caching
doesn't apply — every call re-tokenises the document. A 7B model on
a 50-chunk file is roughly 50 × 30s = 25 min of cold work. Practical
recommendation: run contextualisation against cloud LLMs once during
ingest, then everyday retrieval is local-only as before.

### Validation plan
`rag-evaluation-jp/notebooks/08_contextual_retrieval.ipynb`:
compare baseline vs. contextualised chunks on identical queries.
Use the same eval queries; recompute only the chunk vectors.

---

## 3. Query expansion: HyDE / Multi-Query / Step-Back

**Targets weakness (3)** primarily. All three rewrite the *query side*
before retrieval. They're cheap to try and don't require re-ingest.

### 3a. HyDE — Hypothetical Document Embeddings

Gao, Ma, Lin, Callan — *Precise Zero-Shot Dense Retrieval without
Relevance Labels* (ACL 2023). <https://arxiv.org/abs/2212.10496>

**Mechanism**: ask an LLM to fabricate a plausible answer document for
the query, then embed that and search. The fabricated document is in
the same register as real documents (declarative, full sentences), so
the embedding lands closer to the right neighbourhood than the raw
question would. Hallucinations are fine — only the embedding is used.

```
query
  → LLM "write a 3-sentence document that would answer this"
  → hypothetical_doc
  → encode_query(hypothetical_doc)
  → Qdrant search
```

**Implementation site**: extend `retrieve.retrieve` to take a
`hyde: bool` flag. When set, prepend an `llm.generate` call before
`encode_query`.

**Cost**: +1 LLM call per query (~1-3s with local Qwen, ~0.5s with Gemini
Flash). The retrieve hook currently has no LLM dependency — adding
HyDE would change that, so it should be a per-query opt-in (e.g.,
triggered by query type or by an explicit CLI flag), not the default.

### 3b. Multi-Query Retrieval

**Mechanism**: ask the LLM to paraphrase the query into 3-5 variants,
run each through retrieve, fuse the rankings via RRF (see §1).

**Implementation site**: `retrieve.retrieve(query, multi_query=N)` —
loops over paraphrases, merges with the same RRF helper used for
hybrid retrieval.

**Cost**: 1 LLM call (to generate variants) + N parallel embed + N
parallel Qdrant queries. Worth it when one strong recall + ranking
matters more than minimal latency (planning, postmortem, deep dives).

LangChain has a reference implementation worth reading:
`langchain.retrievers.multi_query.MultiQueryRetriever`.

### 3c. Step-Back Prompting

Zheng et al. (Google DeepMind) — *Take a Step Back: Evoking Reasoning
via Abstraction in Large Language Models* (ICLR 2024).
<https://arxiv.org/abs/2310.06117>

**Mechanism**: rewrite a specific query as a more abstract one before
retrieval, then use both for final retrieval. Pairs well with abstract
documents — the abstract query reaches the abstract corpus level, the
original query keeps the specific signal.

```
specific:  「〇〇社の2023年Q3の売上低下の原因」
  → Step-Back
abstract:  「企業の売上低下を引き起こす一般的な要因は何か」
```

**Implementation site**: similar to Multi-Query but with a different
prompt; retrieve on both specific + abstract, merge.

### 3d. RAG-Fusion (composition of the above)

Rackauckas — *RAG-Fusion: a New Take on Retrieval-Augmented Generation*
(2024). <https://arxiv.org/abs/2402.03367>

Multi-Query + RRF, productionised. Use as a recipe rather than as a
distinct mechanism.

Industry-scale validation:
*Scaling Retrieval Augmented Generation with RAG Fusion: Lessons from
an Industry Deployment* (2026). <https://arxiv.org/abs/2603.02153>
Shows that the recall lift from fusion survives the downstream
re-ranking / dedup / context-window-trimming pipeline — relevant when
considering whether the extra retrieval cost is worth it end-to-end.

---

## 4. Document Summary Index

Optional companion to Contextual Retrieval, often used together.

**Mechanism**: build a secondary index keyed by per-document summaries.
Retrieve summaries first; when a summary matches, fetch the full chunks
of that document. Reduces "topical-but-not-answering" hits (weakness 4)
by gating chunk-level retrieval on document-level relevance.

LlamaIndex has a reference implementation
(`llama_index.indices.document_summary`).

Not as high-priority as §1-3 for agent-kms's typical chunk volumes
(<10k), but worth revisiting once a single source exceeds ~500 chunks.

---

## Implementation order (proposed)

1. **Hybrid retrieval (BM25 + Dense)** — `sparse.py` already exists.
   Wire it in, re-ingest with `--reset`, validate in
   `rag-evaluation-jp/07_hybrid_retrieval.ipynb`. Lowest unknown, highest
   immediate win on identifier queries.
2. **Contextual Retrieval** — biggest single-change impact on abstract
   corpora per Anthropic's benchmark. Needs an LLM-during-ingest path
   that the project doesn't currently have. Plan against cloud-LLM
   cost via prompt caching, or accept a slow one-time ingest for local.
3. **Query expansion (HyDE / Multi-Query / Step-Back)** — only after
   §1 and §2 are in place, since they amplify whatever retriever is
   underneath. Try Step-Back first (single extra LLM call, easy A/B),
   then Multi-Query / HyDE based on which queries still miss.
4. **Document Summary Index** — defer until a single source exceeds
   ~500 chunks; currently no source does.

---

## Validation discipline

Every item above must be:

1. Implemented behind a flag (no behaviour change to existing users by
   default).
2. Compared head-to-head against the current pipeline on at least one
   reproducible corpus (rag-evaluation-jp `EVAL_MODE=public`).
3. Cross-checked on the private corpus via `EVAL_MODE=private`
   (aggregate metrics only — see rag-evaluation-jp README for the
   publishable-vs-withheld breakdown).
4. Documented in this file with the empirical result before becoming a
   default. Negative results stay in the doc — "we tried, it didn't
   help" is as valuable as positive results.

The corpus-size guardrail from earlier discussion still applies:
**meaningful comparison needs ~1000+ chunks**. Below that, ranking
differences between methods sit inside noise.

---

## Status (2026-05-15)

| # | Item | Code stub | Validation | Production |
|---|------|-----------|------------|------------|
| 1 | Hybrid (BM25+Dense) | ✅ `sparse.py` exists, not wired | ❌ | ❌ |
| 2 | Contextual Retrieval | ❌ | ❌ | ❌ |
| 3a | HyDE | ❌ | ❌ | ❌ |
| 3b | Multi-Query | ❌ | ❌ | ❌ |
| 3c | Step-Back | ❌ | ❌ | ❌ |
| 4 | Document Summary | ❌ | ❌ | ❌ |

Last revisited: 2026-05-15.
