"""
Unified SQLite schema shared by all sibling projects.

One open_db() instead of separate schema-creation call sites (ingest.py/
embed.py/index_files.py each used to create its own fragment independently).
No ALTER-TABLE migration shims: existing projects' DBs already have the full
schema below, and any rebuilt-from-scratch DB is created directly against it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    started_at     TEXT,
    ended_at       TEXT,
    title          TEXT,
    cwd            TEXT,
    entrypoint     TEXT,
    version        TEXT,
    git_branch     TEXT,
    message_count  INTEGER,
    excluded       INTEGER NOT NULL DEFAULT 0,
    source         TEXT NOT NULL DEFAULT 'claude',
    source_mtime   INTEGER,
    source_size    INTEGER,
    parent_session_id       TEXT,  -- for source='subagent': owning top-level session_id
    agent_type               TEXT,  -- subagent_type, e.g. 'auditor'
    spawn_depth               INTEGER,
    spawned_from_message_id  TEXT  -- message_id of the parent's Agent tool_use call
);

CREATE TABLE IF NOT EXISTS messages (
    message_id    TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    parent_id     TEXT,
    timestamp     TEXT NOT NULL,
    role          TEXT NOT NULL,
    type          TEXT NOT NULL,
    text_content  TEXT,
    has_tool_use  INTEGER,
    tool_names    TEXT,
    raw_json      TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session   ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_role      ON messages(role);

CREATE TABLE IF NOT EXISTS attachments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id    TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    position      INTEGER,                  -- order within message content
    kind          TEXT NOT NULL,            -- 'image' | 'document' | 'file' | other
    media_type    TEXT,                     -- e.g. 'image/jpeg', 'application/pdf'
    source_type   TEXT,                     -- 'base64' | 'path' | 'url'
    original_path TEXT,                     -- for path-referenced attachments
    archive_path  TEXT,                     -- relative to project root
    sha256        TEXT,
    size_bytes    INTEGER,
    extracted     INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    FOREIGN KEY (message_id) REFERENCES messages(message_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_attachments_message ON attachments(message_id);
CREATE INDEX IF NOT EXISTS idx_attachments_session ON attachments(session_id);
CREATE INDEX IF NOT EXISTS idx_attachments_sha256  ON attachments(sha256);

-- Artifact index: Write/Edit tool calls with file paths
CREATE TABLE IF NOT EXISTS artifacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    message_id      TEXT NOT NULL,
    timestamp       TEXT,
    tool_name       TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    content_preview TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
    FOREIGN KEY (message_id) REFERENCES messages(message_id)
);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_path    ON artifacts(file_path);
CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
    file_path, content_preview,
    content='artifacts',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS artifacts_ai AFTER INSERT ON artifacts BEGIN
    INSERT INTO artifacts_fts(rowid, file_path, content_preview)
    VALUES (new.id, new.file_path, new.content_preview);
END;
CREATE TRIGGER IF NOT EXISTS artifacts_ad AFTER DELETE ON artifacts BEGIN
    INSERT INTO artifacts_fts(artifacts_fts, rowid, file_path, content_preview)
    VALUES('delete', old.id, old.file_path, old.content_preview);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text_content,
    content='messages',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

-- Triggers to keep FTS in sync.
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
"""

FILE_CHUNKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    path        TEXT NOT NULL,
    section     TEXT,
    text        TEXT NOT NULL,
    sha256      TEXT,
    mtime       REAL,
    source_type TEXT,
    weight      REAL DEFAULT 1.0,
    vector      BLOB
);
"""

EMBEDDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    message_id TEXT PRIMARY KEY,
    vector     BLOB NOT NULL,
    FOREIGN KEY (message_id) REFERENCES messages(message_id)
);
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    """Create/open the archive DB with the full unified schema in one place."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(CORE_SCHEMA)
    conn.executescript(FILE_CHUNKS_SCHEMA)
    conn.executescript(EMBEDDINGS_SCHEMA)
    return conn
