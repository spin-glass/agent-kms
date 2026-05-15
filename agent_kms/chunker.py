"""Pluggable chunkers (markdown H2, yaml-per-file).

Each chunker returns ``Iterable[Chunk]`` decoupled from the embedding /
storage layer so callers can mix sources freely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

H2_RE = re.compile(r"^## (.+)$", re.MULTILINE)
H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)


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
    """Yield one chunk per H2 section under ``root``."""
    if not root.exists():
        return
    for md_path in sorted(root.glob(glob)):
        text = md_path.read_text(encoding="utf-8")
        for heading, body in split_markdown_by_h2(text):
            if len(body) < min_chars:
                continue
            yield Chunk(
                source_file=str(md_path),
                heading=heading,
                body=body,
                severity=default_severity,
                applicability=default_applicability,
                confidence=default_confidence,
                source_type=source_type,
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
