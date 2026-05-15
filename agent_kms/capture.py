"""Capture a PR-review lesson (or other hard-won finding) into the knowledge
base as a Markdown file under ``<knowledge_root>/docs/pr-lessons/``.

Motivation
==========
The Stop-hook lesson extractor only sees Claude Code transcripts. Critical
feedback delivered as a GitHub PR comment — and resolved without a chat
session — would otherwise never enter the knowledge base. This command is
the manual capture path: receive a sharp PR comment, type ``agent-kms
capture --title "..."``, paste the body, done.

What it does
============
1. Generate a deterministic slug from the title (Japanese-safe).
2. Render a single Markdown file with YAML frontmatter capturing
   metadata (severity, applicability, tags, optional PR URL).
3. Write it under ``<knowledge_root>/docs/pr-lessons/`` (overridable).
4. Best-effort: run an immediate re-ingest so the lesson is searchable
   on the next ``UserPromptSubmit`` hook firing without manual steps.
5. If the project ``kms.toml`` has no ingest source covering the target
   directory, print a ready-to-paste TOML snippet instead of failing —
   the file is still written so nothing is lost.

The file format works with the existing ``markdown_h2`` chunker: a single
preamble chunk per file (heading = H1 title), or multiple H2 chunks if
the body adds ``##`` sections.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

DEFAULT_DIR_RELATIVE = "docs/pr-lessons"
ALLOWED_SEVERITY = ("critical", "high", "default")
ALLOWED_APPLICABILITY = ("universal", "conditional", "topic-specific")


def slugify(title: str) -> str:
    """Date-prefixed slug. ASCII-only body for cross-FS safety.

    Non-ASCII titles fall back to a short hash so the filename stays
    portable. The full original title is always preserved in the file's
    H1 heading, so no information is lost — the slug is just for sorting
    + uniqueness on disk.
    """
    today = date.today().isoformat()
    ascii_part = re.sub(r"[^a-z0-9-]+", "-", title.lower()).strip("-")[:40]
    if len(ascii_part) < 5:
        # Title is mostly non-ASCII (Japanese etc.) — fall back to a hash.
        h = hashlib.sha256(title.encode("utf-8")).hexdigest()[:8]
        ascii_part = f"lesson-{h}"
    return f"{today}-{ascii_part}"


def read_body(args: argparse.Namespace) -> str:
    """Read body from --body / --editor / stdin, in that order."""
    if args.body:
        return args.body
    if args.editor:
        editor = os.environ.get("EDITOR", "").strip() or "vi"
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(
                "# (PR-lesson body — write your finding here, then save & exit)\n\n"
                "## What I was doing wrong\n\n"
                "## What I should do instead\n\n"
                "## Why it matters\n"
            )
            tmp_path = Path(tmp.name)
        try:
            subprocess.call([*editor.split(), str(tmp_path)])
            return tmp_path.read_text(encoding="utf-8")
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def render(
    title: str,
    body: str,
    severity: str,
    applicability: str,
    tags: list[str],
    source_pr: str,
) -> str:
    """Build the markdown payload with YAML frontmatter."""
    fm_lines = [
        "---",
        f"captured_at: {datetime.now().isoformat(timespec='seconds')}",
        f"severity: {severity}",
        f"applicability: {applicability}",
    ]
    if tags:
        fm_lines.append("tags: [" + ", ".join(tags) + "]")
    if source_pr:
        fm_lines.append(f"source_pr: {source_pr}")
    fm_lines.append("---")
    body_clean = body.strip() or "_(no body)_"
    return "\n".join(fm_lines) + f"\n\n# {title.strip()}\n\n{body_clean}\n"


def write_file(target_dir: Path, slug: str, content: str, force: bool) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{slug}.md"
    if path.exists() and not force:
        # Disambiguate so two captures on the same day don't collide.
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:6]
        path = target_dir / f"{slug}-{h}.md"
    path.write_text(content, encoding="utf-8")
    return path


def check_file_covered(file_path: Path) -> tuple[bool, str]:
    """Return (covered, suggested_toml).

    "Covered" is decided by running the real chunker over the configured
    ingest sources: if at least one chunk's ``source_file`` resolves to the
    same path as ``file_path``, the file will be ingested. This catches
    both directory-containment failures AND glob mismatches (a root="."
    source with glob="AGENTS.md" would "contain" docs/pr-lessons/ but never
    actually ingest a file there).

    ``suggested_toml`` is a paste-ready TOML block to add to kms.toml when
    coverage is missing.
    """
    try:
        from .config import load_config, reset_cache
        from .ingest import collect_chunks

        reset_cache()
        cfg = load_config()
        chunks = collect_chunks(cfg)
    except Exception:
        return False, _suggest_toml()

    target = file_path.resolve()
    for c in chunks:
        try:
            if Path(c.source_file).resolve() == target:
                return True, ""
        except OSError:
            continue

    # Suggest a relative path if file_path lives under knowledge_root.
    rel = None
    try:
        rel = file_path.parent.resolve().relative_to(cfg.knowledge_root.resolve())
    except (ValueError, OSError):
        pass
    return False, _suggest_toml(rel_path=str(rel) if rel else DEFAULT_DIR_RELATIVE)


def _suggest_toml(rel_path: str = DEFAULT_DIR_RELATIVE) -> str:
    return (
        "\n[[ingest.sources]]\n"
        'name = "pr_lessons"\n'
        'kind = "markdown_h2"\n'
        f'roots = ["{rel_path}"]\n'
        'default_severity = "critical"\n'
        'default_applicability = "universal"\n'
        'default_confidence = 1.0\n'
        "min_chars = 50\n"
    )


def reingest_quiet() -> int:
    """Run an in-process ingest. Returns number of upserted chunks (or -1)."""
    try:
        from .config import load_config, reset_cache
        from .ingest import collect_chunks, upsert_chunks
        from .store import ensure_collection

        reset_cache()
        cfg = load_config()
        ensure_collection()
        chunks = collect_chunks(cfg)
        return upsert_chunks(cfg, chunks)
    except Exception as exc:
        print(f"  warning: re-ingest failed ({exc})", file=sys.stderr)
        return -1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Capture a PR-review lesson into the knowledge base."
    )
    ap.add_argument("--title", required=True, help="Short title (becomes H1)")
    ap.add_argument(
        "--severity",
        choices=ALLOWED_SEVERITY,
        default="critical",
        help="Default critical — PR feedback is usually high-signal.",
    )
    ap.add_argument(
        "--applicability",
        choices=ALLOWED_APPLICABILITY,
        default="universal",
    )
    ap.add_argument(
        "--tags",
        default="",
        help="Comma-separated, e.g. 'prisma,import,architecture'",
    )
    ap.add_argument(
        "--source-pr",
        default="",
        help="Optional PR URL or '#NNN' reference.",
    )
    ap.add_argument(
        "--body",
        default="",
        help="Inline body. If absent, --editor or stdin is used.",
    )
    ap.add_argument(
        "--editor",
        action="store_true",
        help="Open $EDITOR to compose the body.",
    )
    ap.add_argument(
        "--target-dir",
        default=None,
        help=(
            f"Destination directory (default: <knowledge_root>/{DEFAULT_DIR_RELATIVE})."
        ),
    )
    ap.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip the immediate re-ingest after writing the file.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite if a file with the same slug already exists today.",
    )
    args = ap.parse_args()

    body = read_body(args)
    if not body.strip():
        print(
            "error: empty body — pass --body, --editor, or pipe via stdin.",
            file=sys.stderr,
        )
        return 2

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    content = render(
        title=args.title,
        body=body,
        severity=args.severity,
        applicability=args.applicability,
        tags=tags,
        source_pr=args.source_pr,
    )

    # Resolve target directory
    if args.target_dir:
        target_dir = Path(args.target_dir).expanduser().resolve()
    else:
        try:
            from .config import load_config, reset_cache

            reset_cache()
            cfg = load_config()
            target_dir = (cfg.knowledge_root / DEFAULT_DIR_RELATIVE).resolve()
        except Exception:
            target_dir = (Path.cwd() / DEFAULT_DIR_RELATIVE).resolve()

    slug = slugify(args.title)
    path = write_file(target_dir, slug, content, force=args.force)
    print(f"  wrote {path}")

    covered, suggestion = check_file_covered(path)
    if not covered:
        print(
            "\n  ! this file is NOT picked up by any ingest source in",
            file=sys.stderr,
        )
        print(
            "    .agent-kms/kms.toml (either the directory is excluded or the",
            file=sys.stderr,
        )
        print(
            "    source's `glob` does not match *.md). Add the following block",
            file=sys.stderr,
        )
        print("    and re-run `agent-kms ingest`:", file=sys.stderr)
        print(suggestion, file=sys.stderr)
        return 0  # File still written; user can fix kms.toml and re-ingest.

    if args.no_ingest:
        print("  (skipping re-ingest per --no-ingest)")
        return 0

    n = reingest_quiet()
    if n >= 0:
        print(f"  re-ingested ({n} chunks total in collection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
