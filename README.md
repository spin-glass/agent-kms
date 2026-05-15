# agent-kms

> **Generic AI agent knowledge management system.**
> Qdrant-backed vector retrieval + Claude Code Stop-hook lesson extraction.
> Loops session-level insights back into future agent sessions so the same
> blind spot does not have to be rediscovered every time.

[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Why

Claude Code (and most coding agents) hold context for one session and lose
it on Stop. A new session re-discovers the same gotchas, hits the same
blind spots. `agent-kms` is the minimal infra to close that loop:

1. **Retrieve** — at the start of each user prompt, surface relevant past
   knowledge (known issues, lessons learned, conventions) automatically.
2. **Extract** — at Stop, mine the session transcript for new lessons via
   an LLM and upsert them to Qdrant.
3. **Loop** — future sessions see those lessons via retrieve, before
   making the same mistake again.

Extracted from a real migration project where this loop measurably cut
session turn counts for repeated symptom classes (a font-mismatch case
study dropped from 10+ turns / 4 failed iterations to 3 turns / 0 failed
iterations after the lesson was upserted to Qdrant).

---

## Install

```bash
# from source (until PyPI release)
pip install git+https://github.com/spin-glass/agent-kms.git

# or with optional LLM providers
pip install 'git+https://github.com/spin-glass/agent-kms.git[gemini]'
pip install 'git+https://github.com/spin-glass/agent-kms.git[anthropic]'
pip install 'git+https://github.com/spin-glass/agent-kms.git[all]'
```

Requires Python ≥ 3.12 and Docker (for Qdrant).

---

## 3-minute quickstart

```bash
# 1. Start Qdrant
#    Simplest option:
docker run -d --name agent-kms-qdrant -p 6333:6333 \
    -v ~/qdrant_data:/qdrant/storage qdrant/qdrant:v1.18.0
#    Or use the reference compose.yaml shipped with the source tree
#    (`examples/compose.yaml` in the cloned repo / sdist).
#    For long-term personal use, manage Qdrant from a dedicated infra
#    directory rather than from inside this library — see
#    ~/dev/infra/local-llm-stack/ (author's setup).

# 2. Scaffold project config
cd /path/to/your/project
agent-kms init --preset general

# 3. (optional) Add LLM key for Stop-hook lesson extraction
cp .env.example .env
$EDITOR .env   # set GEMINI_API_KEY or ANTHROPIC_API_KEY

# 4. Ingest your knowledge base
#    By default looks for docs/knowledge, docs/known-issues,
#    docs/decisions, docs/adr, data/instincts — edit .agent-kms/kms.toml
agent-kms ingest

# 5. Try retrieve
agent-kms retrieve "how should I handle font fallback?"

# 6. (optional) Install Claude Code hooks for auto-retrieve + auto-extract
agent-kms install-hooks
```

Verify everything is reachable:

```bash
agent-kms doctor
```

---

## How it works

### Storage

A single Qdrant collection (default `agent_knowledge`) of 768-dim
`multilingual-e5-base` vectors, with payload:

```text
text, heading, source_file, source_type, severity, applicability, confidence, tags
```

`severity` (`critical` / `high` / `default`) and `applicability`
(`universal` / `conditional` / `topic-specific`) are additive boost
signals applied at retrieve time — curated importance shapes ranking
without requiring an LLM rerank.

### Retrieve

Threshold-based (no top-N cap): every chunk whose
`cosine + severity_boost + applicability_boost ≥ score_threshold` is
returned. The default threshold (0.93) was empirically tuned for
multilingual-e5-base on Japanese-heavy corpora; tune in `kms.toml`.

### Ingest

Driven by `.agent-kms/kms.toml`:

```toml
[[ingest.sources]]
name = "knowledge"
kind = "markdown_h2"            # splits files at H2 headings
roots = ["docs/knowledge"]
default_severity = "default"
default_applicability = "topic-specific"

[[ingest.sources]]
name = "instinct"
kind = "yaml_file"              # one chunk per YAML file
roots = ["data/instincts"]

[file_severity."docs/knowledge/critical-rule.md"]
severity = "critical"
applicability = "universal"
```

### Session lesson extraction

When `agent-kms install-hooks` is run, the Stop hook spawns
`agent-kms extract-lessons` as a detached subprocess on each session end.
It:
1. Reads the last N turns of the transcript JSONL.
2. Sends them to Gemini (or Anthropic) with the preset's lesson prompt.
3. Sanitises against forbidden vocab + duplicate detection (cosine ≥ 0.95).
4. Upserts surviving lessons with `source_type=session_lesson`.

A second pass extracts domain-specific "anti-patterns" with stricter
criteria and higher severity.

### MCP server

`agent-kms serve` exposes `retrieve_for_planning(query, score_threshold)`
over the MCP stdio transport. Register in `~/.claude.json`:

```json
{
  "mcpServers": {
    "agent-kms": { "command": "agent-kms", "args": ["serve"] }
  }
}
```

---

## Presets

| Preset        | When to use                                                |
|---------------|------------------------------------------------------------|
| `general`     | Default. Reads `docs/{knowledge,known-issues,decisions,adr}` + `data/instincts`. English prompts. The prompts include an explicit "match the transcript's primary language" directive so output stays in the user's working language even though the instructions are in English. |

Select via `--preset` flag or `AGENT_KMS_PRESET` env var. To swap to
Japanese (or another) prompts in your project without changing the
library, place override files at `.agent-kms/prompts/*.md` and point
`[prompts] dir = "prompts"` in `kms.toml` — agent-kms falls back to the
general preset's English prompt only when an override is missing.

---

## Configuration reference

| Env / TOML                       | Default                              | Notes                              |
|----------------------------------|--------------------------------------|------------------------------------|
| `QDRANT_URL`                     | `http://localhost:6333`              | Qdrant endpoint                    |
| `AGENT_KMS_PRESET`               | `general`                            | Preset name                        |
| `AGENT_KMS_COLLECTION`           | `agent_knowledge`                    | Qdrant collection                  |
| `AGENT_KMS_KNOWLEDGE_ROOT`       | `cwd`                                | Resolves source roots              |
| `AGENT_KMS_MODEL`                | `intfloat/multilingual-e5-base`      | SentenceTransformer model          |
| `AGENT_KMS_VECTOR_SIZE`          | `768`                                | Must match the model               |
| `GEMINI_API_KEY`                 | (unset)                              | Primary LLM provider               |
| `ANTHROPIC_API_KEY`              | (unset)                              | Fallback LLM provider              |
| `OLLAMA_URL`                     | `http://localhost:11434`             | Local Ollama daemon endpoint       |
| `OLLAMA_MODEL`                   | `qwen2.5:7b`                         | Local model tag                    |
| `RAG_PROVIDER`                   | `auto`                               | `auto` / `gemini` / `haiku` / `ollama` |
| `AGENT_KMS_RETRIEVE_THRESHOLD`   | `0.93`                               | Used by the auto-retrieve hook     |

---

## Capturing PR-review lessons

Stop-hook lesson extraction only sees Claude Code transcripts. Critical
feedback delivered as a GitHub PR comment — and resolved without a chat
session — would otherwise never enter the knowledge base. Use
``agent-kms capture`` for these manually:

```bash
agent-kms capture \
    --title "Prisma client must import from @japan-ai/japan-ai-core-prisma/client" \
    --severity critical \
    --tags "prisma,import,architecture" \
    --source-pr "https://github.com/org/repo/pull/1234" \
    --editor   # opens $EDITOR for the body; or pipe via stdin
```

This writes ``<knowledge_root>/docs/pr-lessons/<date>-<slug>.md`` with
YAML frontmatter and re-ingests automatically. The default severity is
``critical`` so retrieve's severity boost (``+0.05``) lifts these
lessons above default-severity session lessons — repeated mistakes
should never need a third reminder.

Add this source to your project's ``.agent-kms/kms.toml`` once so
captured files are picked up:

```toml
[[ingest.sources]]
name = "pr_lessons"
kind = "markdown_h2"
roots = ["docs/pr-lessons"]
default_severity = "critical"
default_applicability = "universal"
min_chars = 50
```

If the source is missing, ``agent-kms capture`` writes the file but
prints a ready-to-paste snippet on stderr instead of silently failing.

### Capturing whole PRs in one shot

When the review thread is on GitHub (not in a Claude Code transcript),
``agent-kms capture-pr <ref>`` pulls every review body + per-file inline
comment via the ``gh`` CLI and lets you pick which ones to keep:

```bash
agent-kms capture-pr 1234                # bare number, run inside the repo
agent-kms capture-pr https://github.com/org/repo/pull/1234
agent-kms capture-pr org/repo#1234

# Common flags
agent-kms capture-pr 1234 --all                    # skip the picker
agent-kms capture-pr 1234 --only-changes-requested # noise-filtered first pass
agent-kms capture-pr 1234 --inline-only            # per-file comments only
agent-kms capture-pr 1234 --dry-run                # preview without writing
```

Severity defaults track the comment kind:
``CHANGES_REQUESTED`` reviews → ``critical``,
inline comments and ``COMMENTED`` reviews → ``high``,
``APPROVED`` reviews → ``default``.

For inline comments the captured file embeds the path, line number, and
a 12-line diff-hunk excerpt so the retrieve hit later has enough
context to identify what was being changed.

Requires ``gh`` to be installed and authenticated (``brew install gh &&
gh auth login``).

---

## Local LLM (Ollama) for privacy-sensitive setups

When transcripts contain customer data or other information that must not
leave the machine, point lesson extraction at a local Ollama daemon:

```bash
brew install ollama && ollama serve            # in a separate shell
ollama pull qwen2.5:7b                         # or qwen2.5:14b for higher quality

export RAG_PROVIDER=ollama
export OLLAMA_MODEL=qwen2.5:7b
agent-kms doctor                                # verifies daemon + model
```

No code paths send transcripts off the host once `RAG_PROVIDER=ollama` is
set. Embedding and retrieval are already 100% local (a SentenceTransformer
model running on CPU/GPU), so the full Retrieve + Extract loop becomes
offline-capable.

Quality notes: 7B-class models produce valid JSON but may emit thinner
lessons. 14B-class (`qwen2.5:14b`) is the recommended sweet spot on
Apple-Silicon Macs with ≥18 GB unified memory.

---

## Limitations

- The default embedding model (`multilingual-e5-base`) downloads ~1 GB on
  first use. Run `agent-kms doctor` to check the cache.
- **`score_threshold=0.93` is corpus-specific.** It was tuned on the
  original migration project's corpus. On other Japanese technical
  documentation corpora the empirical sweet spot has been observed in
  the **0.80 – 0.85 range** (see [rag-evaluation-jp][rag-eval] for a
  reproducible study on React JP / JS Primer JP). If `retrieve` returns
  zero chunks on your data, sweep `score_threshold` downward in 0.01
  steps starting from 0.85 before assuming the corpus is the problem.
- **Embedding model choice is the largest lever on Japanese corpora.**
  The same study found `cl-nagoya/ruri-v3-310m` (also 768d, drop-in
  swap via `AGENT_KMS_MODEL`) raised F1 by ~80% over `multilingual-e5-base`
  on Japanese-only documentation. agent-kms keeps `multilingual-e5-base`
  as the default for broad applicability; consider switching when your
  corpus is Japanese-heavy.

[rag-eval]: https://github.com/spin-glass/rag-evaluation-jp
- LLM-based lesson extraction is best-effort. Forbidden-vocab filters
  catch obvious sludge but not subtle low-quality output.
- Hook templates assume `agent-kms` is on `$PATH` (pipx / global pip
  install). For venv-pinned installs, edit the templates to call the
  venv's `agent-kms` binary directly.

---

## License

MIT — see [LICENSE](LICENSE).
