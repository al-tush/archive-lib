#!/usr/bin/env python3
"""
Embed transcript messages into the embeddings table.

Processes only messages not yet in embeddings (incremental).
Skips messages with empty text or role=tool_result (low signal/noise ratio).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import ProjectConfig
from .schema import EMBEDDINGS_SCHEMA

# Source-type weights applied at search time (stored here for reference)
ROLE_WEIGHTS = {
    "user": 1.0,
    "assistant": 0.65,
    "tool_result": 0.35,
    "system": 0.30,
}
# Messages shorter than this are skipped (usually tool noise)
MIN_TEXT_LEN = 40


def vec_to_blob(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


def blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def load_pending(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Return (message_id, role, text_content) for messages not yet embedded."""
    rows = conn.execute("""
        SELECT m.message_id, m.role, m.text_content
        FROM messages m
        LEFT JOIN embeddings e ON e.message_id = m.message_id
        WHERE e.message_id IS NULL
          AND m.text_content IS NOT NULL
          AND length(m.text_content) >= ?
          AND m.role != 'tool_result'
        ORDER BY m.timestamp
    """, (MIN_TEXT_LEN,)).fetchall()
    return rows


def embed_batch(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, batch_size=64, show_progress_bar=False,
                        normalize_embeddings=True)


def run(args, model_name: str) -> int:
    conn = sqlite3.connect(args.db)
    conn.executescript(EMBEDDINGS_SCHEMA)

    if args.reset:
        conn.execute("DELETE FROM embeddings")
        conn.commit()
        print("embeddings table cleared")

    pending = load_pending(conn)
    if not pending:
        print("nothing to embed")
        conn.close()
        return 0

    print(f"pending: {len(pending)} messages")
    if args.dry_run:
        conn.close()
        return 0

    print(f"loading model {model_name} …")
    model = SentenceTransformer(model_name)
    print("model loaded")

    batch_size = args.batch
    total = len(pending)
    inserted = 0

    for start in range(0, total, batch_size):
        chunk = pending[start:start + batch_size]
        texts = [row[2] for row in chunk]
        vecs = embed_batch(model, texts)

        conn.executemany(
            "INSERT OR IGNORE INTO embeddings(message_id, vector) VALUES (?, ?)",
            [(row[0], vec_to_blob(vecs[i])) for i, row in enumerate(chunk)]
        )
        conn.commit()
        inserted += len(chunk)
        pct = inserted / total * 100
        print(f"  {inserted}/{total} ({pct:.0f}%)", end="\r", flush=True)

    print(f"\ndone: {inserted} messages embedded")
    conn.close()
    return 0


def main(config: ProjectConfig, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=config.db_path)
    parser.add_argument("--batch", type=int, default=64, help="embedding batch size")
    parser.add_argument("--dry-run", action="store_true", help="count pending, do not embed")
    parser.add_argument("--reset", action="store_true", help="clear embeddings and re-embed all")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 1
    return run(args, config.model_name)
