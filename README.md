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
docker compose -f $(python -c "import agent_kms, pathlib; print(pathlib.Path(agent_kms.__file__).parent.parent / 'compose.yaml')") up -d
#  or simpler:
docker run -d --name agent-kms-qdrant -p 6333:6333 \
    -v ~/qdrant_data:/qdrant/storage qdrant/qdrant

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
| `general`     | Default. Reads `docs/{knowledge,known-issues,decisions,adr}` + `data/instincts`. English prompts. |
| `cocos_unity` | Reference preset extracted from the source migration project (Japanese prompts, file_severity map, gap rules). Copy + edit to model your own preset on. |

Select via `--preset` flag or `AGENT_KMS_PRESET` env var. Override
prompts by placing them under `.agent-kms/prompts/` and pointing
`[prompts] dir = "..."` in `kms.toml`.

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
| `RAG_PROVIDER`                   | `auto`                               | `auto` / `gemini` / `haiku`        |
| `AGENT_KMS_RETRIEVE_THRESHOLD`   | `0.93`                               | Used by the auto-retrieve hook     |

---

## Limitations

- The default embedding model (`multilingual-e5-base`) downloads ~1 GB on
  first use. Run `agent-kms doctor` to check the cache.
- Threshold tuning (0.93) is empirical for one corpus. Drop to ~0.85 for
  smaller / English-only corpora; raise to ~0.95 for stricter matches.
- LLM-based lesson extraction is best-effort. Forbidden-vocab filters
  catch obvious sludge but not subtle low-quality output.
- Hook templates assume `agent-kms` is on `$PATH` (pipx / global pip
  install). For venv-pinned installs, edit the templates to call the
  venv's `agent-kms` binary directly.

---

## License

MIT — see [LICENSE](LICENSE).
