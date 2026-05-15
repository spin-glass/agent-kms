"""Bulk-capture PR-review feedback into the knowledge base.

Wraps ``gh`` (GitHub CLI) so a single ``agent-kms capture-pr <ref>``
command can:

  1. Resolve the PR (by URL, ``#N``, ``owner/repo#N``, or a bare number
     when run inside the repository).
  2. Pull the review body, top-level PR comments, AND per-file inline
     comments.
  3. Show a numbered list and let the user pick which ones to keep, or
     accept all with ``--all``.
  4. For each selection, hand off to :mod:`agent_kms.capture` to write a
     ``docs/pr-lessons/<date>-<slug>.md`` file with appropriate severity:
     CHANGES_REQUESTED → ``critical``, COMMENTED / inline → ``high``,
     APPROVED → ``default`` (rarely captured).

Inline comments carry the file path + line number + the relevant diff
hunk excerpt in the body, so future retrieval surfaces the context as
well as the reviewer's note.

The command does NOT auto-capture by default: the reviewer's full body
list is shown first so the user can drop noise (e.g., "looks good") and
keep only the substantive findings.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Severity policy: how strongly each comment kind is upserted.
SEVERITY_BY_STATE = {
    "CHANGES_REQUESTED": "critical",
    "COMMENTED": "high",
    "APPROVED": "default",
    "DISMISSED": "default",
    # Inline review comments don't carry a state — treated as "high" by
    # default since the reviewer pointed at a specific line.
    "INLINE": "high",
    "PR_COMMENT": "high",
}


def _run_gh(args: list[str]) -> str:
    """Run a ``gh`` subprocess and return stdout. Exits on auth / not-found."""
    try:
        res = subprocess.run(
            ["gh", *args], capture_output=True, text=True, timeout=30
        )
    except FileNotFoundError:
        print(
            "error: `gh` (GitHub CLI) not installed — install via "
            "`brew install gh` and run `gh auth login`.",
            file=sys.stderr,
        )
        sys.exit(127)
    if res.returncode != 0:
        print(f"error: `gh {' '.join(args)}` failed: {res.stderr.strip()}", file=sys.stderr)
        sys.exit(res.returncode)
    return res.stdout


_URL_RE = re.compile(r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<num>\d+)")
_OWNER_REPO_HASH_RE = re.compile(r"(?P<owner>[^/]+)/(?P<repo>[^/]+)#(?P<num>\d+)")


def parse_pr_ref(ref: str) -> dict:
    """Normalise ``ref`` to ``{owner, repo, number}`` or ``{number}``.

    ``gh pr view`` accepts a URL or a bare number when run inside a repo,
    so for the common case we just return the original ref and let gh
    resolve owner/repo. The explicit parse is only used when the caller
    needs to call ``gh api`` (inline comments) where owner/repo are
    required path segments.
    """
    if m := _URL_RE.search(ref):
        return {
            "owner": m.group("owner"),
            "repo": m.group("repo"),
            "number": int(m.group("num")),
            "raw_ref": ref,
        }
    if m := _OWNER_REPO_HASH_RE.fullmatch(ref):
        return {
            "owner": m.group("owner"),
            "repo": m.group("repo"),
            "number": int(m.group("num")),
            "raw_ref": ref,
        }
    if ref.startswith("#"):
        return {"number": int(ref[1:]), "raw_ref": ref}
    if ref.isdigit():
        return {"number": int(ref), "raw_ref": ref}
    raise ValueError(f"unrecognised PR ref: {ref!r}")


def fetch_pr(parsed: dict) -> dict:
    """Fetch the PR (body, reviews, top-level comments) via ``gh pr view``."""
    out = _run_gh(
        [
            "pr",
            "view",
            parsed["raw_ref"],
            "--json",
            "number,title,body,author,baseRefName,headRefName,url,reviews,comments,state",
        ]
    )
    return json.loads(out)


def fetch_inline_comments(parsed: dict, pr_url: str) -> list[dict]:
    """Fetch per-file inline review comments via ``gh api``.

    Requires ``owner`` + ``repo`` + ``number``. If the caller passed only
    a bare ``#N`` we re-derive them from ``pr_url`` (gh always populates
    that field).
    """
    if "owner" not in parsed:
        if m := _URL_RE.search(pr_url):
            parsed = {**parsed, "owner": m.group("owner"), "repo": m.group("repo")}
        else:
            return []
    try:
        out = _run_gh(
            [
                "api",
                f"repos/{parsed['owner']}/{parsed['repo']}/pulls/{parsed['number']}/comments",
                "--paginate",
            ]
        )
        return json.loads(out)
    except SystemExit:
        return []


def normalise_items(pr: dict, inline: list[dict]) -> list[dict]:
    """Flatten reviews / PR-level comments / inline comments into one list.

    Each item gets a short index code: ``r1``, ``r2`` for reviews,
    ``c1`` for top-level PR comments, ``i1`` for inline comments. The
    code is what the user types when picking interactively.
    """
    items: list[dict] = []
    for i, rev in enumerate(pr.get("reviews") or [], start=1):
        body = (rev.get("body") or "").strip()
        if not body:
            continue
        items.append(
            {
                "code": f"r{i}",
                "kind": "review",
                "state": rev.get("state", "COMMENTED"),
                "author": (rev.get("author") or {}).get("login", "?"),
                "body": body,
                "path": None,
                "line": None,
                "diff_hunk": None,
                "url": rev.get("url") or pr.get("url", ""),
            }
        )
    for i, c in enumerate(pr.get("comments") or [], start=1):
        body = (c.get("body") or "").strip()
        if not body:
            continue
        items.append(
            {
                "code": f"c{i}",
                "kind": "comment",
                "state": "PR_COMMENT",
                "author": (c.get("author") or {}).get("login", "?"),
                "body": body,
                "path": None,
                "line": None,
                "diff_hunk": None,
                "url": c.get("url") or pr.get("url", ""),
            }
        )
    for i, ic in enumerate(inline or [], start=1):
        body = (ic.get("body") or "").strip()
        if not body:
            continue
        items.append(
            {
                "code": f"i{i}",
                "kind": "inline",
                "state": "INLINE",
                "author": (ic.get("user") or {}).get("login", "?"),
                "body": body,
                "path": ic.get("path"),
                "line": ic.get("line") or ic.get("original_line"),
                "diff_hunk": ic.get("diff_hunk"),
                "url": ic.get("html_url"),
            }
        )
    return items


def filter_items(
    items: list[dict],
    only_changes_requested: bool = False,
    inline_only: bool = False,
) -> list[dict]:
    out = items
    if only_changes_requested:
        out = [x for x in out if x["state"] == "CHANGES_REQUESTED"]
    if inline_only:
        out = [x for x in out if x["kind"] == "inline"]
    return out


def print_summary(pr: dict, items: list[dict]) -> None:
    print(f"PR #{pr['number']}: {pr['title']}")
    print(f"  author: {(pr.get('author') or {}).get('login', '?')}  state: {pr.get('state', '?')}")
    print(f"  url:    {pr.get('url', '')}")
    print(f"  items:  {len(items)}\n")
    for x in items:
        head = f"  [{x['code']}] ({x['state']} by @{x['author']})"
        if x["kind"] == "inline":
            head += f"  @ {x['path']}:{x['line']}"
        body_first = " ".join(x["body"].split())[:140]
        print(head)
        print(f"        {body_first}")


def prompt_selection(items: list[dict]) -> list[dict]:
    """Ask the user which items to capture. Returns the picked items."""
    if not items:
        return []
    print("\nSelect items to capture: comma-separated codes (e.g. r1,i3), "
          "'all', or 'q' to quit.")
    raw = input("> ").strip()
    if not raw or raw.lower() == "q":
        return []
    if raw.lower() == "all":
        return items
    wanted = {p.strip() for p in raw.split(",") if p.strip()}
    return [x for x in items if x["code"] in wanted]


def build_capture_args(
    item: dict, pr: dict, default_tags: list[str]
) -> tuple[str, str, str, list[str]]:
    """Return (title, body, severity, tags) for a captured item."""
    severity = SEVERITY_BY_STATE.get(item["state"], "high")

    # Title auto-generation: prefer the inline file path; fall back to
    # the first sentence of the body. Truncate aggressively so the slug
    # generator does not produce a 60-char filename.
    if item["kind"] == "inline" and item.get("path"):
        first_sentence = re.split(r"[。.!?\n]", item["body"], maxsplit=1)[0].strip()
        title = f"{item['path']}: {first_sentence[:80]}"
    else:
        first_sentence = re.split(r"[。.!?\n]", item["body"], maxsplit=1)[0].strip()
        title = first_sentence[:100] or f"PR #{pr['number']} review comment"

    # Body composition. For inline comments, prepend the file:line + a
    # short diff-hunk excerpt so retrieve hits surface the context.
    body_parts: list[str] = []
    if item["kind"] == "inline":
        body_parts.append(f"**ファイル**: `{item['path']}:{item['line']}`")
        if item.get("diff_hunk"):
            hunk = "\n".join(item["diff_hunk"].splitlines()[-12:])
            body_parts.append(f"**該当 diff (末尾 12 行)**:\n```diff\n{hunk}\n```")
    body_parts.append(f"**レビュアー**: @{item['author']} ({item['state']})")
    body_parts.append("")
    body_parts.append("## 指摘内容")
    body_parts.append(item["body"])
    body = "\n\n".join(body_parts)

    tags = list(default_tags)
    if item["kind"] == "inline":
        tags.append("inline")
        if item.get("path"):
            tags.append(Path(item["path"]).suffix.lstrip(".") or "no-ext")
    if "pr-review" not in tags:
        tags.insert(0, "pr-review")

    return title, body, severity, tags


def capture_item(
    item: dict, pr: dict, dry_run: bool, default_tags: list[str], target_dir: str | None
) -> int:
    """Hand off to ``agent-kms capture``. Returns its exit code."""
    title, body, severity, tags = build_capture_args(item, pr, default_tags)
    if dry_run:
        print(f"\n--- [dry-run] would capture [{item['code']}] ---")
        print(f"  title:    {title[:100]}")
        print(f"  severity: {severity}")
        print(f"  tags:     {','.join(tags)}")
        print(f"  source:   {item['url']}")
        return 0

    cmd = [
        "agent-kms", "capture",
        "--title", title,
        "--severity", severity,
        "--tags", ",".join(tags),
        "--source-pr", item["url"] or pr.get("url", ""),
        "--no-ingest",  # ingest once at the end instead of per-comment
    ]
    if target_dir:
        cmd += ["--target-dir", target_dir]
    res = subprocess.run(cmd, input=body, text=True, capture_output=True)
    if res.returncode != 0:
        print(f"  ✗ [{item['code']}] capture failed: {res.stderr.strip()}", file=sys.stderr)
        return res.returncode
    print(f"  ✓ [{item['code']}] {(res.stdout or '').strip()}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Capture PR-review feedback (reviews + inline comments) into agent-kms."
    )
    ap.add_argument("ref", help="PR ref: URL, owner/repo#N, #N, or bare number when run in repo")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Capture every comment without interactive selection.",
    )
    ap.add_argument(
        "--only-changes-requested",
        action="store_true",
        help="Only reviews whose state is CHANGES_REQUESTED.",
    )
    ap.add_argument(
        "--inline-only",
        action="store_true",
        help="Only per-file inline review comments (skip top-level review bodies).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which items would be captured without writing files.",
    )
    ap.add_argument(
        "--tags",
        default="",
        help="Extra comma-separated tags appended to every captured item.",
    )
    ap.add_argument(
        "--target-dir",
        default=None,
        help="Override the target directory for captured markdown files.",
    )
    args = ap.parse_args()

    parsed = parse_pr_ref(args.ref)
    pr = fetch_pr(parsed)
    inline = fetch_inline_comments(parsed, pr.get("url", ""))
    items = normalise_items(pr, inline)
    items = filter_items(
        items,
        only_changes_requested=args.only_changes_requested,
        inline_only=args.inline_only,
    )

    if not items:
        print("(no capturable review comments after filters)")
        return 0

    print_summary(pr, items)

    selected = items if args.all else prompt_selection(items)
    if not selected:
        print("\n  nothing selected.")
        return 0

    extra_tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    print(f"\ncapturing {len(selected)} item(s)...")
    for x in selected:
        capture_item(x, pr, args.dry_run, extra_tags, args.target_dir)

    if args.dry_run:
        return 0

    # Single re-ingest at the end so the embedding model only loads once
    # even for large multi-comment captures.
    print("\nre-ingesting...")
    from .capture import reingest_quiet

    n = reingest_quiet()
    if n >= 0:
        print(f"  ({n} chunks total in collection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
