#!/usr/bin/env python3
"""
Tests for the source-namespace + fork-detection changes in ingest.py/schema.py
(тз_rag_мультиисточник_и_форки_сессий.md in the apteki project). Covers the
plan-of-verification points from that ТЗ (§10):

  1. Schema: two messages with the same message_id but different source insert
     fine; same message_id + same source collides (composite PK works).
  2. Fork detection: two files sharing ~98% of message_ids link via
     session_links and are not duplicated in `messages`.
  3. Below-threshold overlap: NOT treated as a fork — both sessions indexed
     independently, no session_links row.

Run: python3 tests/test_ingest.py [-v]   (from archive-lib/)
"""

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from archive_lib.schema import open_db
from archive_lib.ingest import ingest_file, FORK_OVERLAP_THRESHOLD


def _write_jsonl(path: Path, session_id: str, message_uuids: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, uuid in enumerate(message_uuids):
            event = {
                "type": "user",
                "uuid": uuid,
                "sessionId": session_id,
                "timestamp": f"2026-09-{(i % 27) + 1:02d}T10:00:00.000Z",
                "cwd": "/tmp/project",
                "message": {"role": "user", "content": [{"type": "text", "text": f"message {i}"}]},
            }
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


class _TmpDbTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.conn = open_db(self.root / "transcripts.db")

    def tearDown(self):
        self.conn.close()
        self.tmpdir.cleanup()

    def _ingest(self, session_id: str, uuids: list[str]) -> dict:
        path = self.root / f"{session_id}.jsonl"
        _write_jsonl(path, session_id, uuids)
        return ingest_file(path, self.conn, copy_dest=None, files_dest=None,
                           project_root=self.root, dry_run=False)


class TestCompositePrimaryKey(unittest.TestCase):
    """§10 point 1: same message_id, different source must not collide;
    same message_id, same source must collide (the composite PK doing its job)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.conn = open_db(Path(self.tmpdir.name) / "transcripts.db")

    def tearDown(self):
        self.conn.close()
        self.tmpdir.cleanup()

    def _insert(self, message_id: str, session_id: str, source: str) -> None:
        self.conn.execute(
            "INSERT INTO sessions (session_id, source) VALUES (?, ?)",
            (session_id, source),
        )
        self.conn.execute(
            """INSERT INTO messages
               (message_id, session_id, timestamp, role, type, source)
               VALUES (?, ?, '2026-09-26T00:00:00Z', 'user', 'user', ?)""",
            (message_id, session_id, source),
        )

    def test_same_id_different_source_ok(self):
        self._insert("shared-uuid", "session-a", "claude")
        self._insert("shared-uuid", "session-b", "copilot")  # must not raise
        rows = self.conn.execute(
            "SELECT source, session_id FROM messages WHERE message_id = ? ORDER BY source",
            ("shared-uuid",),
        ).fetchall()
        self.assertEqual(rows, [("claude", "session-a"), ("copilot", "session-b")])

    def test_same_id_same_source_collides(self):
        self._insert("dup-uuid", "session-a", "claude")
        with self.assertRaises(sqlite3.IntegrityError):
            self._insert("dup-uuid", "session-b", "claude")


class TestForkDetection(_TmpDbTestCase):
    """§10 point 2: synthetic A (100 msgs) + B (98 shared + 2 new, overlap 98%)."""

    def test_fork_linked_and_not_duplicated(self):
        shared = [f"msg-{i}" for i in range(98)]
        a_only = ["a-extra-1", "a-extra-2"]
        b_only = ["b-extra-1", "b-extra-2"]

        stats_a = self._ingest("session-parent", shared + a_only)
        self.assertEqual(stats_a["messages"], 100)

        stats_b = self._ingest("session-child", shared + b_only)
        # Only the 2 genuinely new messages get inserted under the child session.
        self.assertEqual(stats_b["messages"], 2)

        link = self.conn.execute(
            """SELECT parent_session_id, link_type, overlap_ratio FROM session_links
               WHERE child_session_id = ?""",
            ("session-child",),
        ).fetchone()
        self.assertIsNotNone(link, "session_links must record the fork")
        self.assertEqual(link[0], "session-parent")
        self.assertEqual(link[1], "background_fork")
        self.assertAlmostEqual(link[2], 98 / 100, places=3)

        total_messages = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        self.assertEqual(total_messages, 102, "no duplicate rows for the 98 shared ids")

        # The shared ids stay attributed to the parent session, not duplicated
        # under the child.
        owners = {row[0] for row in self.conn.execute(
            "SELECT DISTINCT session_id FROM messages WHERE message_id IN "
            + "(" + ",".join("?" * len(shared)) + ")", shared
        ).fetchall()}
        self.assertEqual(owners, {"session-parent"})

    def test_ingesting_same_file_twice_stays_idempotent(self):
        """Re-running ingest on the SAME session_id (unchanged file re-ingested,
        e.g. --force) must not be misread as a fork of itself."""
        ids = [f"msg-{i}" for i in range(10)]
        self._ingest("session-x", ids)
        stats = self._ingest("session-x", ids)  # re-ingest, same session_id
        self.assertEqual(stats["messages"], 10)
        links = self.conn.execute(
            "SELECT COUNT(*) FROM session_links WHERE child_session_id = 'session-x'"
        ).fetchone()[0]
        self.assertEqual(links, 0)


class TestBelowThresholdOverlap(_TmpDbTestCase):
    """§10 point 3: overlap below FORK_OVERLAP_THRESHOLD is NOT treated as a
    known fork pattern — no session_links row, and it is not silently
    IGNOREd either (§6 point 4). Within the SAME source, a genuine id
    collision that isn't an actual fork is exactly the kind of unforeseen
    anomaly §8's file-level try/except (in main(), not ingest_file() itself)
    exists to isolate — ingest_file() must let it fail loudly rather than
    quietly dropping or merging data it can't attribute correctly."""

    def test_no_fork_but_raises_loudly_on_real_collision(self):
        self.assertLess(0.40, FORK_OVERLAP_THRESHOLD)
        shared = [f"msg-{i}" for i in range(40)]
        a_only = [f"a-only-{i}" for i in range(60)]
        b_only = [f"b-only-{i}" for i in range(60)]

        stats_a = self._ingest("session-a", shared + a_only)
        self.assertEqual(stats_a["messages"], 100)

        with self.assertRaises(sqlite3.IntegrityError):
            self._ingest("session-b", shared + b_only)

        self.conn.rollback()
        link = self.conn.execute(
            "SELECT 1 FROM session_links WHERE child_session_id = 'session-b'"
        ).fetchone()
        self.assertIsNone(link, "below-threshold overlap must not create a session_links row")

    def test_below_threshold_across_different_sources_is_harmless(self):
        """Cross-source overlap (a different source's ids coincidentally
        matching) never collides — the composite PK keys on (source,
        message_id), and detect_fork() only ever compares within the same
        source — so this is fully independent regardless of overlap ratio."""
        from archive_lib.ingest import detect_fork

        same_looking_ids = [f"msg-{i}" for i in range(40)]
        self.conn.execute("INSERT INTO sessions (session_id, source) VALUES ('claude-sess', 'claude')")
        self.conn.executemany(
            """INSERT INTO messages (message_id, session_id, timestamp, role, type, source)
               VALUES (?, 'claude-sess', '2026-09-26T00:00:00Z', 'user', 'user', 'claude')""",
            [(mid,) for mid in same_looking_ids],
        )
        self.conn.commit()

        parent, ratio = detect_fork(self.conn, "copilot", "some-copilot-session",
                                    set(same_looking_ids))
        self.assertIsNone(parent, "a different source must never match as a fork parent")
        self.assertEqual(ratio, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
