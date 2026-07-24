#!/usr/bin/env python3
"""
Full update pipeline: ingest new transcripts + rebuild RAG indexes.

Steps:
  1. ingest.py         — Claude Code JSONL → transcripts.db + archive/transcripts/
  2. ingest_copilot.py — GitHub Copilot sessions → transcripts.db  (skip if no chatSessions dir)
  3. embed.py          — new messages → embeddings table
  4. index_files.py    — changed project/memory files → file_chunks table
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from .config import ProjectConfig


def run_step(label: str, cmd: list[str]) -> bool:
    """Run a subprocess step. Returns True on success."""
    print(f"\n{'='*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"{'='*60}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] '{label}' exited with code {result.returncode}", file=sys.stderr)
    return result.returncode == 0


def main(config: ProjectConfig, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="pass --dry-run to each step (no writes)")
    parser.add_argument("--no-copilot", action="store_true",
                        help="skip ingest_copilot step")
    args = parser.parse_args(argv)

    scripts = config.scripts_dir
    python = sys.executable
    dry = ["--dry-run"] if args.dry_run else []
    steps_ok = 0
    steps_failed = 0

    ok = run_step(
        "ingest.py — Claude Code sessions",
        [python, str(scripts / "ingest.py")] + dry,
    )
    steps_ok += ok
    steps_failed += not ok

    copilot_dir = config.copilot_chatsessions_dir
    if args.no_copilot:
        print("\n[skip] ingest_copilot.py — --no-copilot")
    elif copilot_dir is None:
        print("\n[skip] ingest_copilot.py — no Copilot workspace configured for this project")
    elif not copilot_dir.is_dir():
        print(f"\n[skip] ingest_copilot.py — chatSessions not found:\n       {copilot_dir}")
    else:
        ok = run_step(
            "ingest_copilot.py — GitHub Copilot sessions",
            [python, str(scripts / "ingest_copilot.py")] + dry,
        )
        steps_ok += ok
        steps_failed += not ok

    ok = run_step(
        "embed.py — transcript embeddings",
        [python, str(scripts / "embed.py")] + dry,
    )
    steps_ok += ok
    steps_failed += not ok

    ok = run_step(
        "index_files.py — project/memory file chunks",
        [python, str(scripts / "index_files.py")] + dry,
    )
    steps_ok += ok
    steps_failed += not ok

    print(f"\n{'='*60}", flush=True)
    status = "OK" if steps_failed == 0 else f"{steps_failed} FAILED"
    print(f"  done: {steps_ok} ok, {steps_failed} failed  [{status}]", flush=True)
    print(f"{'='*60}\n", flush=True)

    return 0 if steps_failed == 0 else 1
