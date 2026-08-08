#!/usr/bin/env python3
"""
Index project markdown files into file_chunks table.

Chunking strategy: split by ## headers (parent sections), then sub-split
paragraphs if section > MAX_CHUNK_WORDS. Chunks < MIN_CHUNK_CHARS are skipped.

Incremental: each file is re-indexed only when its mtime changes.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import ProjectConfig
from .hybrid_words import normalize_hybrid_words
from .schema import FILE_CHUNKS_SCHEMA

MIN_CHUNK_CHARS = 80
MAX_CHUNK_WORDS = 400


def vec_to_blob(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalize_frontmatter(text: str) -> str:
    """Convert YAML frontmatter to plain text, preserving semantic content.

    Removes YAML delimiters and structural keys (metadata, type, node_type),
    but keeps name/description values as readable text so embeddings capture them.

    Before:                          After:
      ---                            Не предлагать мелкие задачи
      name: Не предлагать мелкие     У пользователя паттерн...
      description: У пользователя    (rest of file)
      metadata:
        type: feedback
      ---
    """
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text

    frontmatter = text[3:end]
    body = text[end + 4:].lstrip("\n")

    kept_lines = []
    skip_keys = {"metadata", "type", "node_type", "  type", "properties"}
    for line in frontmatter.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(k + ":") for k in skip_keys):
            continue
        if ":" in stripped:
            value = stripped.split(":", 1)[1].strip()
            if value:
                kept_lines.append(value)
        else:
            kept_lines.append(stripped)

    header = "\n".join(kept_lines)
    return (header + "\n\n" + body).strip() if header else body


def split_into_chunks(text: str) -> list[tuple[str | None, str]]:
    """Split markdown text into (section_header, chunk_text) pairs."""
    section_pattern = re.compile(r'^#{1,3}\s+.+', re.MULTILINE)
    matches = list(section_pattern.finditer(text))

    if not matches:
        return [(None, chunk) for chunk in _split_paragraphs(text)]

    parts: list[tuple[str | None, str]] = []
    boundaries = [m.start() for m in matches] + [len(text)]

    preamble = text[:boundaries[0]].strip()
    if len(preamble) >= MIN_CHUNK_CHARS:
        parts.extend((None, c) for c in _split_paragraphs(preamble))

    for i, m in enumerate(matches):
        header = m.group(0).strip()
        section_text = text[m.end():boundaries[i + 1]].strip()
        if not section_text:
            continue
        sub_chunks = _split_paragraphs(section_text)
        parts.extend((header, f"{header}\n\n{c}") for c in sub_chunks)

    return parts


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraph-sized chunks, merging short ones."""
    raw = re.split(r'\n{2,}', text.strip())
    chunks: list[str] = []
    current = ""

    for para in raw:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) < MAX_CHUNK_WORDS * 5:  # ~5 chars/word estimate
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if len(current) >= MIN_CHUNK_CHARS:
                chunks.append(current)
            current = para

    if len(current) >= MIN_CHUNK_CHARS:
        chunks.append(current)

    return chunks if chunks else [text.strip()]


def get_file_mtime(path: Path) -> float:
    return path.stat().st_mtime


def load_indexed_mtimes(conn: sqlite3.Connection) -> dict[str, float]:
    rows = conn.execute(
        "SELECT DISTINCT path, mtime FROM file_chunks"
    ).fetchall()
    result: dict[str, float] = {}
    for path, mtime in rows:
        result[path] = max(result.get(path, 0.0), mtime)
    return result


def resolve_path(rel_path: str, root_key: str | None,
                 project_root: Path, memory_root: Path | None) -> Path:
    if root_key == "memory":
        if memory_root is None:
            raise ValueError(
                f"file_registry entry {rel_path!r} has root='memory' but this "
                "project's config.memory_root is None"
            )
        return memory_root / rel_path
    return project_root / rel_path


def index_file(conn: sqlite3.Connection, model: SentenceTransformer,
               rel_path: str, source_type: str, weight: float,
               root_key: str | None, project_root: Path, memory_root: Path | None) -> int:
    full_path = resolve_path(rel_path, root_key, project_root, memory_root)
    if not full_path.exists():
        print(f"  skip (not found): {rel_path}")
        return 0

    mtime = get_file_mtime(full_path)
    text = normalize_frontmatter(full_path.read_text(encoding="utf-8"))
    text = normalize_hybrid_words(text, warn_context=rel_path)
    chunks = split_into_chunks(text)
    valid = [(sec, c) for sec, c in chunks if len(c) >= MIN_CHUNK_CHARS]

    if not valid:
        return 0

    texts = [c for _, c in valid]
    vecs = model.encode(texts, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False)

    conn.execute("DELETE FROM file_chunks WHERE path = ?", (rel_path,))

    conn.executemany(
        """INSERT INTO file_chunks(path, section, text, sha256, mtime, source_type, weight, vector)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (rel_path, sec, c, sha256_text(c), mtime, source_type, weight, vec_to_blob(vecs[i]))
            for i, (sec, c) in enumerate(valid)
        ]
    )
    conn.commit()
    return len(valid)


def run(args, config: ProjectConfig) -> int:
    conn = sqlite3.connect(args.db)
    conn.executescript(FILE_CHUNKS_SCHEMA)

    if args.list:
        rows = conn.execute(
            "SELECT path, COUNT(*) as n, source_type, weight FROM file_chunks GROUP BY path ORDER BY path"
        ).fetchall()
        if not rows:
            print("(no files indexed)")
        for path, n, st, w in rows:
            print(f"  {n:3} chunks  w={w:.2f}  [{st}]  {path}")
        conn.close()
        return 0

    if args.reset:
        conn.execute("DELETE FROM file_chunks")
        conn.commit()
        print("file_chunks table cleared")

    indexed_mtimes = {} if args.reset else load_indexed_mtimes(conn)

    to_index = []
    for entry in config.file_registry:
        rel_path, source_type, weight, root_key = entry
        full_path = resolve_path(rel_path, root_key, config.project_root, config.memory_root)
        if not full_path.exists():
            continue
        current_mtime = get_file_mtime(full_path)
        if rel_path not in indexed_mtimes or indexed_mtimes[rel_path] < current_mtime:
            to_index.append((rel_path, source_type, weight, root_key))

    if not to_index:
        print("all files up to date")
        conn.close()
        return 0

    print(f"files to index: {len(to_index)}")
    if args.dry_run:
        for rel_path, st, w, rk in to_index:
            root_label = "[memory]" if rk == "memory" else "[project]"
            print(f"  {root_label} [{st} w={w}] {rel_path}")
        conn.close()
        return 0

    print(f"loading model {config.model_name} …")
    model = SentenceTransformer(config.model_name)
    print("model loaded\n")

    total_chunks = 0
    for rel_path, source_type, weight, root_key in to_index:
        n = index_file(conn, model, rel_path, source_type, weight, root_key,
                       config.project_root, config.memory_root)
        root_label = "[M]" if root_key == "memory" else "   "
        print(f"  {n:3} chunks  {root_label} [{source_type}]  {rel_path}")
        total_chunks += n

    print(f"\ndone: {total_chunks} chunks across {len(to_index)} files")
    conn.close()
    return 0


def main(config: ProjectConfig, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=config.db_path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--list", action="store_true", help="list indexed files")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 1
    return run(args, config)
