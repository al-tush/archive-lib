#!/usr/bin/env python3
"""One-off migration: retroactively strip already-extracted base64 image/document
data from JSONL transcript copies committed before write_compacted_copy (see
ingest.py) learned to strip base64 at write time.

Source of truth for "already extracted": the attachments table, populated by a
prior `ingest.py` run (source_type='base64', extracted=1, archive_path set —
meaning the actual bytes already live under archive/files/, deduped by sha256).
A block is only rewritten if a matching (message_id, position) row is found
there; anything not yet extracted, or extraction that failed, is left alone.

Safe for the same reason described in ingest.strip_base64_blocks: neither the
FTS/embedding ingest path nor restore.py ever reads block.source.data for
image/document blocks, and restore.py already has a source.type == "file"
branch that renders the path instead of decoding base64.

Kept as a separate script (not a mode of ingest.py) because it is a batch
file-rewrite operation over already-committed data — the kind of change that
should be run, checked with `git diff --stat` on a sample, and only then
trusted at full scale (see project rule: verify a script's output on real data
as soon as the idea comes up, not after a full run).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .ingest import strip_base64_blocks


def load_lookup(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        """SELECT message_id, position, media_type, archive_path, sha256, size_bytes
           FROM attachments
           WHERE source_type = 'base64' AND extracted = 1 AND archive_path IS NOT NULL"""
    ).fetchall()
    return {
        (message_id, position): {
            "media_type": media_type, "archive_path": archive_path,
            "sha256": sha256, "size_bytes": size_bytes,
        }
        for message_id, position, media_type, archive_path, sha256, size_bytes in rows
    }


def process_file(path: Path, lookup: dict, dry_run: bool) -> tuple[bool, int, int]:
    """Returns (changed, original_size, new_size)."""
    original_size = path.stat().st_size
    out_lines: list[str] = []
    changed = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                out_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            if isinstance(event, dict) and strip_base64_blocks(event, lookup):
                changed = True
            out_lines.append(json.dumps(event, ensure_ascii=False) + "\n")
    if not changed:
        return False, original_size, original_size
    new_content = "".join(out_lines)
    new_size = len(new_content.encode("utf-8"))
    if not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(new_content, encoding="utf-8")
        tmp.replace(path)
    return True, original_size, new_size


def main(db_path: Path, transcripts_dir: Path, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing anything")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N matching files (0 = all) — "
                             "for checking a sample before a full run")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(str(db_path))
    lookup = load_lookup(conn)
    conn.close()
    if not args.quiet:
        print(f"Loaded {len(lookup)} extracted base64 attachment(s) from {db_path}.")

    files = sorted(transcripts_dir.rglob("*.jsonl"))
    if args.limit:
        files = files[:args.limit]

    files_touched = 0
    bytes_before = 0
    bytes_after = 0
    for path in files:
        changed, before, after = process_file(path, lookup, args.dry_run)
        if not changed:
            continue
        files_touched += 1
        bytes_before += before
        bytes_after += after
        if not args.quiet:
            print(f"  {path.relative_to(transcripts_dir.parent)}: {before:,} -> {after:,} bytes")

    if not args.quiet:
        saved = bytes_before - bytes_after
        verb = "Would touch" if args.dry_run else "Touched"
        print(f"\n{verb} {files_touched}/{len(files)} file(s), saved {saved:,} bytes "
              f"({bytes_before:,} -> {bytes_after:,}).")
    return 0


if __name__ == "__main__":
    sys.exit(1)  # invoke via a project's archive/scripts/ wrapper, not directly
