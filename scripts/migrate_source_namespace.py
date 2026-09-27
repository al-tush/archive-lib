#!/usr/bin/env python3
"""
One-time migration for an existing transcripts.db: add messages.source
(composite PK (source, message_id) — see тз_rag_мультиисточник_и_форки_сессий.md
in the apteki project for the full rationale), attachments.source,
artifacts.source, and the session_links table.

schema.py's open_db() intentionally has no ALTER-TABLE migration shims — this
script is the one-time bridge for a DB created before this change. It is
idempotent (skips the schema step if messages.source already exists) and
always backs up the DB file first unless --no-backup is passed.

Usage:
  python3 migrate_source_namespace.py --db /path/to/project/archive/transcripts.db
  python3 migrate_source_namespace.py --db ... --backfill-links   # also retroactively
                                                                    # detect forks in
                                                                    # already-ingested history
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from archive_lib.ingest import FORK_OVERLAP_THRESHOLD  # noqa: E402


def already_migrated(conn: sqlite3.Connection) -> bool:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    return "source" in cols


def migrate_messages(conn: sqlite3.Connection) -> None:
    """Rebuild messages with source + composite PK, preserving rowid so the
    existing messages_fts (content_rowid='rowid') index stays aligned without
    a full rebuild."""
    conn.execute("""
        CREATE TABLE messages_new (
            message_id    TEXT NOT NULL,
            session_id    TEXT NOT NULL,
            parent_id     TEXT,
            timestamp     TEXT NOT NULL,
            role          TEXT NOT NULL,
            type          TEXT NOT NULL,
            text_content  TEXT,
            has_tool_use  INTEGER,
            tool_names    TEXT,
            raw_json      TEXT,
            source        TEXT NOT NULL DEFAULT 'claude',
            PRIMARY KEY (source, message_id),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    conn.execute("""
        INSERT INTO messages_new
            (rowid, message_id, session_id, parent_id, timestamp, role, type,
             text_content, has_tool_use, tool_names, raw_json, source)
        SELECT m.rowid, m.message_id, m.session_id, m.parent_id, m.timestamp, m.role, m.type,
               m.text_content, m.has_tool_use, m.tool_names, m.raw_json,
               COALESCE(s.source, 'claude')
        FROM messages m
        LEFT JOIN sessions s ON s.session_id = m.session_id
    """)
    conn.execute("DROP TABLE messages")
    conn.execute("ALTER TABLE messages_new RENAME TO messages")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session   ON messages(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_role      ON messages(role)")
    # DROP TABLE drops the old table's triggers with it — recreate them.
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, text_content) VALUES (new.rowid, new.text_content);
        END;
        CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, text_content) VALUES('delete', old.rowid, old.text_content);
        END;
        CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts, rowid, text_content) VALUES('delete', old.rowid, old.text_content);
            INSERT INTO messages_fts(rowid, text_content) VALUES (new.rowid, new.text_content);
        END;
    """)


def migrate_attachments_and_artifacts(conn: sqlite3.Connection) -> None:
    for table in ("attachments", "artifacts"):
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if "source" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN source TEXT NOT NULL DEFAULT 'claude'")
        conn.execute(f"""
            UPDATE {table} SET source = COALESCE(
                (SELECT s.source FROM sessions s WHERE s.session_id = {table}.session_id),
                'claude'
            )
        """)


def create_session_links(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS session_links (
            child_session_id  TEXT NOT NULL,
            parent_session_id TEXT NOT NULL,
            source            TEXT NOT NULL DEFAULT 'claude',
            link_type         TEXT NOT NULL CHECK (link_type IN
                                ('background_fork', 'resume', 'cross_source_reference')),
            overlap_ratio     REAL,
            detected_at       TEXT NOT NULL,
            PRIMARY KEY (child_session_id, parent_session_id, source)
        );
        CREATE INDEX IF NOT EXISTS idx_session_links_parent ON session_links(parent_session_id);
    """)


def backfill_links(conn: sqlite3.Connection, detected_at: str) -> int:
    """Retroactively apply ingest.py's fork-detection heuristic across all
    already-loaded (non-subagent) sessions, oldest first — a fork's parent
    must already be examined/known before its child is checked against it,
    which started_at ASC order guarantees for the real background-fork case
    (parent always started earlier)."""
    n = 0
    sessions = conn.execute(
        "SELECT session_id, source FROM sessions WHERE source != 'subagent' ORDER BY started_at ASC"
    ).fetchall()
    for session_id, source in sessions:
        ids = {row[0] for row in conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ? AND source = ?",
            (session_id, source),
        ).fetchall()}
        if not ids:
            continue
        placeholders = ",".join("?" * len(ids))
        candidates = [r[0] for r in conn.execute(
            f"""SELECT DISTINCT session_id FROM messages
                WHERE source = ? AND session_id != ? AND message_id IN ({placeholders})""",
            (source, session_id, *ids),
        ).fetchall()]
        best_parent, best_ratio = None, 0.0
        for cand in candidates:
            count = conn.execute(
                f"""SELECT COUNT(*) FROM messages
                    WHERE source = ? AND session_id = ? AND message_id IN ({placeholders})""",
                (source, cand, *ids),
            ).fetchone()[0]
            ratio = count / len(ids)
            if ratio > best_ratio:
                best_ratio, best_parent = ratio, cand
        if best_parent is not None and best_ratio >= FORK_OVERLAP_THRESHOLD:
            conn.execute(
                """INSERT OR REPLACE INTO session_links
                   (child_session_id, parent_session_id, source, link_type, overlap_ratio, detected_at)
                   VALUES (?, ?, ?, 'background_fork', ?, ?)""",
                (session_id, best_parent, source, best_ratio, detected_at),
            )
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, required=True, help="Path to transcripts.db")
    parser.add_argument("--backfill-links", action="store_true",
                        help="Also retroactively detect forks across already-ingested history")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip the .bak copy (default: always back up first)")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    try:
        if already_migrated(conn):
            print("Already migrated (messages.source exists) — schema step skipped.")
        else:
            if not args.no_backup:
                backup = args.db.with_name(args.db.name + f".bak-{int(time.time())}")
                shutil.copy2(args.db, backup)
                print(f"Backed up to {backup}")
            migrate_messages(conn)
            migrate_attachments_and_artifacts(conn)
            conn.commit()
            print("Schema migrated: messages/attachments/artifacts.source.")

        create_session_links(conn)  # idempotent — always ensure it exists
        conn.commit()

        if args.backfill_links:
            detected_at = datetime.now(timezone.utc).isoformat()
            n = backfill_links(conn, detected_at)
            conn.commit()
            print(f"Backfilled {n} fork link(s) into session_links.")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
