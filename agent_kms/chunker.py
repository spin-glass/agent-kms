"""Pluggable chunkers (markdown H2, yaml-per-file).

Each chunker returns ``Iterable[Chunk]`` decoupled from the embedding /
storage layer so callers can mix sources freely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

H2_RE = re.compile(r"^## (.+)$", re.MULTILINE)
H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)

# Matches a leading YAML frontmatter block: '---\n<yaml>\n---\n' at file start.
# Captured group 1 is the YAML body so the caller can ``yaml.safe_load`` it.
_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


@dataclass
class Chunk:
    source_file: str  # relative path (str) for stable IDs across machines
    heading: str
    body: str
    severity: str = "default"
    applicability: str = "topic-specific"
    confidence: float = 1.0
    source_type: str = "knowledge"
    tags: list[str] = field(default_factory=list)
    # Optional frontmatter-sourced extras. Empty defaults so legacy chunks
    # without a frontmatter block round-trip unchanged.
    source_pr: str = ""
    captured_at: str = ""

    @property
    def embed_text(self) -> str:
        """Heading-prepend embedding text. Empirically improves MRR.

        See README for the experiment that justifies this transformation.
        """
        stem = Path(self.source_file).stem
        return f"{stem} | {self.heading}\n\n{self.body}"


def split_markdown_by_h2(text: str) -> list[tuple[str, str]]:
    """Return list of (heading, body) for each H2 section.

    Files with no H2 produce a single chunk for the whole file. The preamble
    before the first H2 is emitted as a chunk whose heading is the H1 title
    (or ``"(preamble)"`` if missing).
    """
    matches = list(H2_RE.finditer(text))
    chunks: list[tuple[str, str]] = []

    if not matches:
        h1 = H1_RE.search(text)
        heading = h1.group(1).strip() if h1 else "(no-heading)"
        body = text.strip()
        if body:
            chunks.append((heading, body))
        return chunks

    preamble = text[: matches[0].start()].strip()
    if preamble:
        h1 = H1_RE.search(preamble)
        heading = h1.group(1).strip() if h1 else "(preamble)"
        chunks.append((heading, preamble))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            chunks.append((m.group(1).strip(), section))

    return chunks


def _parse_markdown_frontmatter(text: str) -> tuple[dict, str]:
    """Strip a leading YAML frontmatter block. Returns ``(meta, body)``.

    A file without a frontmatter block returns ``({}, text)`` unchanged, so
    callers can blindly chain this in front of existing chunking logic.
    Malformed YAML is treated as "no frontmatter" (silent) — partial files
    should not crash an ingest run.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    return meta, text[m.end():]


def _coerce_tags(value: object) -> list[str]:
    """YAML tags can be either a list (canonical) or a comma string. Normalise."""
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    return []


def chunk_markdown_h2(
    root: Path,
    glob: str = "*.md",
    *,
    min_chars: int = 100,
    default_severity: str = "default",
    default_applicability: str = "topic-specific",
    default_confidence: float = 1.0,
    source_type: str = "knowledge",
) -> Iterable[Chunk]:
    """Yield one chunk per H2 section under ``root``.

    A leading YAML frontmatter block is parsed once per file and its values
    override the per-source defaults for every chunk derived from the file.
    Recognised keys: ``severity``, ``applicability``, ``confidence``,
    ``tags``, ``source_pr``, ``captured_at``. Unknown keys are ignored
    (forward-compatible).

    File-level overrides via ``kms.toml`` ``[file_severity]`` still apply
    on top of frontmatter — they run later in ``ingest._apply_file_overrides``.
    """
    if not root.exists():
        return
    for md_path in sorted(root.glob(glob)):
        text = md_path.read_text(encoding="utf-8")
        meta, body = _parse_markdown_frontmatter(text)

        severity = str(meta.get("severity") or default_severity)
        applicability = str(meta.get("applicability") or default_applicability)
        try:
            confidence = float(meta.get("confidence", default_confidence))
        except (TypeError, ValueError):
            confidence = default_confidence
        tags = _coerce_tags(meta.get("tags"))
        source_pr = str(meta.get("source_pr") or "")
        # PyYAML auto-parses unquoted ISO timestamps to ``datetime`` and the
        # default ``str()`` then loses the canonical "T" separator. Use
        # ``isoformat()`` when available so payload values stay sortable
        # ``YYYY-MM-DDTHH:MM:SS`` strings regardless of how the value was
        # written in the frontmatter.
        captured_raw = meta.get("captured_at")
        if captured_raw is None or captured_raw == "":
            captured_at = ""
        elif hasattr(captured_raw, "isoformat"):
            captured_at = captured_raw.isoformat()
        else:
            captured_at = str(captured_raw)

        for heading, body_text in split_markdown_by_h2(body):
            if len(body_text) < min_chars:
                continue
            yield Chunk(
                source_file=str(md_path),
                heading=heading,
                body=body_text,
                severity=severity,
                applicability=applicability,
                confidence=confidence,
                source_type=source_type,
                tags=list(tags),  # copy so per-chunk edits don't leak
                source_pr=source_pr,
                captured_at=captured_at,
            )


def chunk_yaml_per_file(
    root: Path,
    glob: str = "*.yaml",
    *,
    default_severity: str = "default",
    default_applicability: str = "topic-specific",
    default_confidence: float = 1.0,
    source_type: str = "knowledge",
) -> Iterable[Chunk]:
    """Yield one chunk per YAML file under ``root``.

    Heading is the file stem; useful for "instinct"-style YAML where each
    file is one self-contained piece of knowledge.
    """
    if not root.exists():
        return
    for yaml_path in sorted(root.glob(glob)):
        text = yaml_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        yield Chunk(
            source_file=str(yaml_path),
            heading=yaml_path.stem,
            body=text,
            severity=default_severity,
            applicability=default_applicability,
            confidence=default_confidence,
            source_type=source_type,
        )


CHUNKERS = {
    "markdown_h2": chunk_markdown_h2,
    "yaml_file": chunk_yaml_per_file,
}
