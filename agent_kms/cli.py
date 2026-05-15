"""agent-kms CLI — single entry point for ingest / retrieve / serve / hooks.

Subcommands:
  init             Scaffold ``.agent-kms/kms.toml`` in CWD from a preset.
  ingest           Run config-driven ingest into Qdrant.
  retrieve         Run a one-off retrieve and print results.
  serve            Start the MCP server (FastMCP stdio transport).
  extract-lessons  Extract lessons from a transcript file → Qdrant.
  postmortem       Extract long-window postmortem from a transcript.
  improve          Run auto-improve gap detection on a transcript.
  install-hooks    Copy Claude Code hook templates into a target repo.
  doctor           Health check: Qdrant reachable, model cached, keys set.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser().resolve()
    kms_dir = target / ".agent-kms"
    kms_dir.mkdir(parents=True, exist_ok=True)
    kms_toml = kms_dir / "kms.toml"
    if kms_toml.exists() and not args.force:
        print(f"refusing to overwrite {kms_toml} (use --force)", file=sys.stderr)
        return 1

    from . import config as cfg_mod

    src = cfg_mod.PRESETS_DIR / args.preset / "kms.toml"
    if not src.exists():
        print(f"unknown preset {args.preset!r}", file=sys.stderr)
        return 1
    shutil.copy(src, kms_toml)

    env_example = target / ".env.example"
    package_env_example = Path(__file__).resolve().parent.parent / ".env.example"
    if package_env_example.exists() and not env_example.exists():
        shutil.copy(package_env_example, env_example)

    print(f"wrote {kms_toml}")
    print(f"wrote {env_example} (template)" if env_example.exists() else "")
    print()
    print("Next steps:")
    print("  1. Start Qdrant:   docker run -d -p 6333:6333 -v ~/qdrant_data:/qdrant/storage qdrant/qdrant")
    print("  2. Copy .env.example to .env and fill in keys (or skip — Qdrant works without LLM keys).")
    print("  3. agent-kms ingest")
    print("  4. agent-kms retrieve 'your query'")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from .ingest import main as ingest_main

    sys.argv = ["ingest"] + (["--reset"] if args.reset else [])
    if args.preset:
        sys.argv += ["--preset", args.preset]
    return ingest_main()


def _cmd_retrieve(args: argparse.Namespace) -> int:
    from .retrieve import retrieve

    results = retrieve(args.query, score_threshold=args.threshold)
    if args.json:
        import json as _json

        print(_json.dumps(results, ensure_ascii=False))
        return 0
    if not results:
        print("(no chunks above threshold)")
        return 0
    for r in results:
        src = r.get("source", "?")
        heading = r.get("heading", "")
        score = r.get("score", 0.0)
        sev = r.get("severity", "")
        app = r.get("applicability", "")
        text = " ".join(r.get("text", "").split())[:240]
        print(f"  • [{r.get('source_type', '?')} {sev}/{app}] {src} #{heading} (score={score:.3f})")
        print(f"    {text}")
        print()
    return 0


def _cmd_serve(_: argparse.Namespace) -> int:
    from .server import main as serve_main

    serve_main()
    return 0


def _cmd_extract_lessons(args: argparse.Namespace) -> int:
    from .session_extract import main as extract_main

    sys.argv = ["extract-lessons", "--transcript", args.transcript]
    if args.tail_turns:
        sys.argv += ["--tail-turns", str(args.tail_turns)]
    return extract_main()


def _cmd_postmortem(args: argparse.Namespace) -> int:
    from .session_postmortem import main as pm_main

    sys.argv = ["postmortem", "--transcript", args.transcript]
    return pm_main()


def _cmd_improve(args: argparse.Namespace) -> int:
    from .auto_improve import main as improve_main

    sid = args.session_id or Path(args.transcript).stem
    sys.argv = ["improve", "--transcript", args.transcript, "--session-id", sid]
    return improve_main()


def _cmd_effectiveness(args: argparse.Namespace) -> int:
    from .effectiveness import main as effectiveness_main

    sid = args.session_id or Path(args.transcript).stem
    sys.argv = ["effectiveness", "--transcript", args.transcript, "--session-id", sid]
    return effectiveness_main()


def _cmd_capture_pr(args: argparse.Namespace) -> int:
    from .capture_pr import main as capture_pr_main

    sys.argv = ["capture-pr", args.ref]
    if args.all:
        sys.argv += ["--all"]
    if args.only_changes_requested:
        sys.argv += ["--only-changes-requested"]
    if args.inline_only:
        sys.argv += ["--inline-only"]
    if args.dry_run:
        sys.argv += ["--dry-run"]
    if args.tags:
        sys.argv += ["--tags", args.tags]
    if args.target_dir:
        sys.argv += ["--target-dir", args.target_dir]
    return capture_pr_main()


def _cmd_capture(args: argparse.Namespace) -> int:
    from .capture import main as capture_main

    sys.argv = ["capture", "--title", args.title]
    if args.severity:
        sys.argv += ["--severity", args.severity]
    if args.applicability:
        sys.argv += ["--applicability", args.applicability]
    if args.tags:
        sys.argv += ["--tags", args.tags]
    if args.source_pr:
        sys.argv += ["--source-pr", args.source_pr]
    if args.body:
        sys.argv += ["--body", args.body]
    if args.editor:
        sys.argv += ["--editor"]
    if args.target_dir:
        sys.argv += ["--target-dir", args.target_dir]
    if args.no_ingest:
        sys.argv += ["--no-ingest"]
    if args.force:
        sys.argv += ["--force"]
    return capture_main()


def _cmd_install_hooks(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser().resolve()
    hooks_dir = target / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent.parent / "scripts" / "hook-templates"
    if not src.exists():
        print(f"hook templates not found at {src}", file=sys.stderr)
        return 1
    n = 0
    for hook in src.glob("*.sh"):
        dst = hooks_dir / hook.name
        if dst.exists() and not args.force:
            print(f"  skipping existing {dst} (use --force)")
            continue
        shutil.copy(hook, dst)
        dst.chmod(0o755)
        print(f"  installed {dst}")
        n += 1
    print(f"installed {n} hook(s)")
    print()
    print("Register in .claude/settings.json under hooks:")
    print('  "UserPromptSubmit": [{"command": ".claude/hooks/auto-rag-retrieve.sh"}]')
    print('  "Stop":             [{"command": ".claude/hooks/extract-session-lessons.sh"}]')
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    import socket
    from urllib.parse import urlparse

    print("agent-kms doctor")
    ok = True

    # 1. Qdrant
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    parsed = urlparse(url)
    host, port = parsed.hostname or "localhost", parsed.port or 6333
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"  ✓ Qdrant reachable at {url}")
    except OSError as e:
        print(f"  ✗ Qdrant NOT reachable at {url}: {e}")
        ok = False

    # 2. Embedding model cache
    from .store import MODEL_NAME

    cache = Path.home() / ".cache" / "huggingface" / "hub"
    flag = cache / f"models--{MODEL_NAME.replace('/', '--')}"
    if flag.exists():
        print(f"  ✓ Embedding model cached: {MODEL_NAME}")
    else:
        print(f"  ⚠ Embedding model NOT cached (first ingest will download ~1GB): {MODEL_NAME}")

    # 3. API keys
    gem = bool(os.environ.get("GEMINI_API_KEY", "").strip()) and not os.environ.get(
        "GEMINI_API_KEY", ""
    ).startswith("AIzaSy.")
    ant = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()) and not os.environ.get(
        "ANTHROPIC_API_KEY", ""
    ).startswith("sk-ant-...")
    if gem:
        print("  ✓ GEMINI_API_KEY set")
    if ant:
        print("  ✓ ANTHROPIC_API_KEY set")

    # 4. Ollama (only probe if user opted in)
    rag_provider = os.environ.get("RAG_PROVIDER", "auto").lower()
    ollama_opted_in = (
        rag_provider == "ollama"
        or os.environ.get("RAG_PROVIDER_FALLBACK", "").lower() == "ollama"
        or bool(os.environ.get("OLLAMA_URL", "").strip())
        or bool(os.environ.get("OLLAMA_MODEL", "").strip())
    )
    if ollama_opted_in:
        from urllib.error import URLError
        from urllib.request import Request, urlopen

        ourl = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
        omodel = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        try:
            req = Request(f"{ourl}/api/tags", method="GET")
            with urlopen(req, timeout=2) as resp:
                import json as _json

                tags = _json.loads(resp.read().decode("utf-8")).get("models", [])
                names = {t.get("name", "") for t in tags}
                if any(omodel == n or n.startswith(omodel + ":") for n in names):
                    print(f"  ✓ Ollama reachable at {ourl}, model {omodel!r} pulled")
                else:
                    print(
                        f"  ⚠ Ollama reachable at {ourl} but model {omodel!r} not "
                        f"pulled — run `ollama pull {omodel}`"
                    )
        except (URLError, OSError, TimeoutError) as e:
            print(f"  ✗ Ollama NOT reachable at {ourl}: {e}")
            ok = False

    if not gem and not ant and not ollama_opted_in:
        print("  ⚠ No LLM provider set — Stop-hook lesson extraction will be skipped")

    # 5. Config
    try:
        from .config import load_config

        cfg = load_config()
        print(f"  ✓ Config loaded: preset={cfg.preset} collection={cfg.collection}")
        print(f"    knowledge_root: {cfg.knowledge_root}")
        print(f"    sources:        {[s.name for s in cfg.sources]}")
    except Exception as e:
        print(f"  ✗ Config error: {e}")
        ok = False

    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent-kms", description="Generic AI agent KMS")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="Scaffold .agent-kms/kms.toml in CWD")
    pi.add_argument("--target", default=".", help="Target project directory")
    pi.add_argument("--preset", default="general", help="Preset name")
    pi.add_argument("--force", action="store_true", help="Overwrite existing file")
    pi.set_defaults(func=_cmd_init)

    pin = sub.add_parser("ingest", help="Ingest sources into Qdrant")
    pin.add_argument("--preset", default=None)
    pin.add_argument("--reset", action="store_true")
    pin.set_defaults(func=_cmd_ingest)

    pr = sub.add_parser("retrieve", help="One-off retrieve")
    pr.add_argument("query")
    pr.add_argument("--threshold", type=float, default=0.93)
    pr.add_argument(
        "--json",
        action="store_true",
        help="Output the raw retrieve result list as JSON (machine-readable).",
    )
    pr.set_defaults(func=_cmd_retrieve)

    ps = sub.add_parser("serve", help="Start FastMCP server")
    ps.set_defaults(func=_cmd_serve)

    pe = sub.add_parser("extract-lessons", help="Stop-hook lesson extraction")
    pe.add_argument("--transcript", required=True)
    pe.add_argument("--tail-turns", type=int, default=30)
    pe.set_defaults(func=_cmd_extract_lessons)

    pp = sub.add_parser("postmortem", help="Stop-hook long-window postmortem")
    pp.add_argument("--transcript", required=True)
    pp.set_defaults(func=_cmd_postmortem)

    pa = sub.add_parser("improve", help="Auto-improve gap detection")
    pa.add_argument("--transcript", required=True)
    pa.add_argument(
        "--session-id",
        default=None,
        help="Override session_id (defaults to transcript filename stem).",
    )
    pa.set_defaults(func=_cmd_improve)

    pe2 = sub.add_parser(
        "effectiveness",
        help="Summarise which retrieved chunks the assistant referenced this session",
    )
    pe2.add_argument("--transcript", required=True)
    pe2.add_argument(
        "--session-id",
        default=None,
        help="Override session_id (defaults to transcript filename stem).",
    )
    pe2.set_defaults(func=_cmd_effectiveness)

    pcpr = sub.add_parser(
        "capture-pr",
        help=(
            "Capture review comments from a GitHub PR into the knowledge base "
            "(requires the `gh` CLI authenticated)."
        ),
    )
    pcpr.add_argument("ref", help="PR ref: URL, owner/repo#N, #N, or bare number")
    pcpr.add_argument("--all", action="store_true",
                      help="Capture every comment without interactive selection.")
    pcpr.add_argument("--only-changes-requested", action="store_true")
    pcpr.add_argument("--inline-only", action="store_true")
    pcpr.add_argument("--dry-run", action="store_true")
    pcpr.add_argument("--tags", default="")
    pcpr.add_argument("--target-dir", default=None)
    pcpr.set_defaults(func=_cmd_capture_pr)

    pc = sub.add_parser(
        "capture",
        help="Capture a PR-review lesson into the knowledge base as a Markdown file.",
    )
    pc.add_argument("--title", required=True)
    pc.add_argument("--severity", default=None, choices=["critical", "high", "default"])
    pc.add_argument(
        "--applicability",
        default=None,
        choices=["universal", "conditional", "topic-specific"],
    )
    pc.add_argument("--tags", default="")
    pc.add_argument("--source-pr", default="")
    pc.add_argument("--body", default="")
    pc.add_argument("--editor", action="store_true")
    pc.add_argument("--target-dir", default=None)
    pc.add_argument("--no-ingest", action="store_true")
    pc.add_argument("--force", action="store_true")
    pc.set_defaults(func=_cmd_capture)

    ph = sub.add_parser("install-hooks", help="Copy Claude Code hook templates")
    ph.add_argument("--target", default=".", help="Target project directory")
    ph.add_argument("--force", action="store_true")
    ph.set_defaults(func=_cmd_install_hooks)

    pd = sub.add_parser("doctor", help="Health check")
    pd.set_defaults(func=_cmd_doctor)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
