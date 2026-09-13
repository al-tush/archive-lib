#!/usr/bin/env python3
"""
Search the transcripts archive built by ingest.py.

By default searches only role=user messages (direct user input — highest signal).
Use --role/--roles to override, or --all-roles to search everything.

FTS5 query syntax: AND, OR, NOT, NEAR, prefix*, "exact phrase".
Russian: no stemming — bare Cyrillic words are auto-expanded with a trailing *
to catch inflections (e.g. "тревога" -> "тревога*"). Use --no-expand to disable.
Use --hybrid for best thematic recall.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .config import ProjectConfig
from .pyenv import ensure_semantic_capable as _ensure_semantic_capable
from .registry import KNOWN_ARCHIVES
from .timeutil import LOCAL_TZ, LOCAL_TZ_LABEL

_STALE_SECONDS = 3600  # re-index if last run was more than 1 hour ago


def _auto_update_if_stale(config: ProjectConfig, db: Path) -> None:
    """Run ingest.py / ingest_copilot.py (via the project's own thin wrapper
    scripts) if there are unindexed sessions OR last indexing was >1 hour ago."""
    source_dirs = [config.claude_source_dir] + list(config.extra_claude_source_dirs)
    copilot_source_dir = config.copilot_chatsessions_dir

    last_indexed_file = db.parent / ".last_indexed"
    last_indexed = 0.0
    if last_indexed_file.exists():
        try:
            last_indexed = float(last_indexed_file.read_text().strip())
        except Exception:
            pass

    time_stale = (time.time() - last_indexed) > _STALE_SECONDS

    known_claude: set[str] = set()
    known_copilot: set[str] = set()
    if db.exists():
        try:
            conn = sqlite3.connect(db)
            known_claude = {row[0] for row in conn.execute(
                "SELECT session_id FROM sessions WHERE source='claude' OR source IS NULL"
            )}
            known_copilot = {row[0] for row in conn.execute(
                "SELECT session_id FROM sessions WHERE source='copilot'"
            )}
            conn.close()
        except Exception:
            pass

    new_claude: list[Path] = []
    for source_dir in source_dirs:
        if source_dir.is_dir():
            new_claude += [p for p in source_dir.glob("*.jsonl") if p.stem not in known_claude]

    new_copilot: list[Path] = []
    if copilot_source_dir is not None and copilot_source_dir.is_dir():
        new_copilot = [p for p in copilot_source_dir.glob("*.jsonl") if p.stem not in known_copilot]

    if not new_claude and not new_copilot and not time_stale:
        return

    reason = []
    if new_claude:
        reason.append(f"{len(new_claude)} new Claude session(s)")
    if new_copilot:
        reason.append(f"{len(new_copilot)} new Copilot session(s)")
    if time_stale:
        mins = int((time.time() - last_indexed) / 60)
        reason.append(f"last indexed {mins}m ago")
    print(f"[auto-update] {', '.join(reason)} — updating FTS index...", file=sys.stderr)

    scripts = config.scripts_dir
    all_ok = True

    if any(d.is_dir() for d in source_dirs) and (new_claude or time_stale):
        result = subprocess.run(
            [sys.executable, str(scripts / "ingest.py"), "--quiet"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[auto-update] ingest.py failed:\n{result.stderr[:400]}", file=sys.stderr)
            all_ok = False

    if copilot_source_dir is not None and copilot_source_dir.is_dir() and (new_copilot or time_stale):
        result = subprocess.run(
            [sys.executable, str(scripts / "ingest_copilot.py"), "--quiet"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"[auto-update] ingest_copilot.py failed:\n{result.stderr[:400]}", file=sys.stderr)
            all_ok = False

    if all_ok:
        print("[auto-update] done.", file=sys.stderr)


_FTS_OPS = frozenset({"AND", "OR", "NOT", "NEAR"})


def _expand_cyrillic(query: str) -> str:
    """Add * to bare Cyrillic words in an FTS query.

    Quoted phrases are passed through unchanged. FTS operators (AND/OR/NOT/NEAR) are preserved.
    Words already suffixed with * are left as-is.

    Example: 'просыпался утром' → 'просыпался* утром*'
    """
    result: list[str] = []
    i = 0
    while i < len(query):
        ch = query[i]
        if ch == '"':
            end = query.find('"', i + 1)
            if end == -1:
                result.append(query[i:])
                break
            result.append(query[i:end + 1])
            i = end + 1
            continue
        if ch in '()':
            result.append(ch)
            i += 1
            continue
        m = re.match(r'[а-яёА-ЯЁ]+', query[i:])
        if m:
            word = m.group(0)
            i += len(word)
            has_star = i < len(query) and query[i] == '*'
            if has_star:
                i += 1
            if word.upper() in _FTS_OPS:
                result.append(word + ('*' if has_star else ''))
                continue
            result.append(word + '*')
            continue
        result.append(ch)
        i += 1
    return ''.join(result)


_FTS_UNSAFE_BARE_CHAR = re.compile(r'[^\w*() \t\n]', re.UNICODE)


def _sanitize_fts_query(query: str) -> str:
    """Replace characters outside quoted phrases that the FTS5 query parser
    can't handle as bare tokens (e.g. '/', '.') with spaces.

    Without this, a query like 'Sect.dev устроился' or 'отказа / планы'
    reaches sqlite3 as-is and MATCH raises 'fts5: syntax error near "."'
    instead of returning results. Quoted phrases are passed through
    unchanged — FTS5 tokenizes their contents as a literal phrase, so stray
    punctuation inside quotes doesn't trigger a parser error.

    Example: 'Sect.dev устроился' -> 'Sect dev устроился'
    """
    result: list[str] = []
    i = 0
    while i < len(query):
        ch = query[i]
        if ch == '"':
            end = query.find('"', i + 1)
            if end == -1:
                result.append(query[i:])
                break
            result.append(query[i:end + 1])
            i = end + 1
            continue
        if _FTS_UNSAFE_BARE_CHAR.match(ch):
            result.append(' ')
            i += 1
            continue
        result.append(ch)
        i += 1
    return ''.join(result)


_CODE_LINE_STARTS = ("#!/", "import ", "from ", "def ", "class ", "return ", "if __name__")
_LINE_NUM_RE = re.compile(r"^\s*\d+\t")  # cat-n line number prefix: "  42\t"


def _looks_like_code(text: str) -> bool:
    """True if the message content looks like source code.

    Handles content formatted with line numbers (Read/cat-n format: "1\tdef foo():").
    """
    if not text:
        return False
    lines = text[:1000].splitlines()
    stripped = [_LINE_NUM_RE.sub("", ln) for ln in lines]
    if stripped and stripped[0].lstrip().startswith("#!/"):
        return True
    code_lines = sum(1 for ln in stripped if ln.strip().startswith(_CODE_LINE_STARTS))
    return code_lines >= 3


_RAG_ECHO_HEAD_PREFIXES = ("[expand] ", "[auto-update]", "(no results)", "═", "loading model")
_RAG_ECHO_MARKERS = (
    "archive/scripts/query.py", "archive/scripts/restore.py", "archive/scripts/eval.py",
    "Recall@", "MRR@", "  HIT@", "  MISS ",
)


def _looks_like_rag_echo(text: str) -> bool:
    """True if the message content is this tool's own command/output rather
    than user or biographical content (query.py/restore.py invocations,
    '[expand]'/'[auto-update]' banners, empty-result footers, eval.py/embed.py
    run output).

    Live sessions get indexed while still running, so without this filter a
    follow-up query in the same session routinely finds its own prior
    query.py call (or another session's stderr banner) as a top "result" —
    self-referential noise that crowds out real content. eval.py output is a
    separate case worth naming: a `... | tail -N` capture of its per-query
    HIT@/MISS report (or the final Recall@/MRR@ line) drops the "loading
    model" header the head-prefix check relies on, so those need their own
    markers rather than relying on head position.
    """
    if not text:
        return False
    head = text[:200].lstrip()
    if head.startswith(_RAG_ECHO_HEAD_PREFIXES):
        return True
    return any(marker in text[:1000] for marker in _RAG_ECHO_MARKERS)


# Role weights for final score calculation
ROLE_WEIGHTS = {
    "user": 1.0,
    "assistant": 0.65,
    "tool_result": 0.35,
    "system": 0.30,
}
# Time-decay half-life in days: score halves every N days
TIME_DECAY_HALFLIFE = 180.0


def _time_decay(timestamp: Optional[str]) -> float:
    if not timestamp:
        return 1.0
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days = max(0.0, (now - ts).total_seconds() / 86400)
        return math.exp(-math.log(2) * days / TIME_DECAY_HALFLIFE)
    except Exception:
        return 1.0


def _cosine(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b))  # vectors are pre-normalized


@functools.lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Cached: a single query.py invocation may call this many times (once per
    eval.py golden-set query, or once per --archives leg), and reloading the
    SentenceTransformer each time is the dominant cost by orders of magnitude."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _blob_to_vec(b: bytes):
    import numpy as np
    return np.frombuffer(b, dtype=np.float32)


def _dedup_by_text(results: list[dict]) -> list[dict]:
    """Drop results whose text_content duplicates an earlier (higher-ranked) result.

    The same file/message content routinely gets indexed under multiple
    message_ids or sessions (e.g. a file Read into several Claude sessions),
    so dedup by message_id/chunk_id alone (see _merge_pools) lets identical
    passages appear more than once in one result list.
    """
    seen: set[str] = set()
    deduped = []
    for r in results:
        text = (r.get("text_content") or "").strip()
        key = re.sub(r"\s+", " ", text).lower()
        if key:
            if key in seen:
                continue
            seen.add(key)
        deduped.append(r)
    return deduped


def _merge_pools(pools: dict, k: int, guarantees: dict[str, int]) -> list[dict]:
    """Merge scored pools with guaranteed minimum slots, then fill by score.

    `guarantees` gives a minimum number of guaranteed slots per pool name —
    prevents high-volume corpora (e.g. diary) from drowning out sparse ones
    (e.g. memory). Each project supplies its own via config.pool_guarantees.
    """
    result: list[dict] = []
    seen: set = set()
    remaining = {name: list(items) for name, items in pools.items()}

    # Phase 1: guaranteed slots from each pool
    for pool_name, min_slots in guarantees.items():
        pool = remaining.get(pool_name, [])
        for item in pool[:min_slots]:
            uid = item.get("message_id") or item.get("chunk_id")
            if uid not in seen:
                result.append(item)
                seen.add(uid)
        remaining[pool_name] = pool[min_slots:]

    # Phase 2: fill remaining slots with best scores across all pools
    leftover = sorted(
        [item for pool in remaining.values() for item in pool],
        key=lambda x: x["score"], reverse=True,
    )
    for item in leftover:
        if len(result) >= k:
            break
        uid = item.get("message_id") or item.get("chunk_id")
        if uid not in seen:
            result.append(item)
            seen.add(uid)

    result.sort(key=lambda x: x["score"], reverse=True)
    return result[:k]


def fmt_ts(ts: str) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M") + f" {LOCAL_TZ_LABEL}"
    except ValueError:
        return ts[:16]


def local_bound_to_utc(s: str, end_of_day: bool) -> str:
    """Локальную границу --from/--to (YYYY-MM-DD[THH:MM:SS]) → UTC-строка для
    сравнения с сырым timestamp в БД (он в UTC). Показ локальный (MSK), поэтому
    и границы дат трактуем как локальные, иначе день съезжает на 3 часа."""
    if "T" not in s:
        s = s + ("T23:59:59" if end_of_day else "T00:00:00")
    try:
        dt = datetime.fromisoformat(s).replace(tzinfo=LOCAL_TZ)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return s


def short(session_id: str) -> str:
    return (session_id or "")[:8]


def manage_exclusion(conn: sqlite3.Connection, session_prefix: str, exclude: bool) -> None:
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT session_id, title FROM sessions WHERE session_id LIKE ? || '%'",
        (session_prefix,)
    ).fetchall()
    if not rows:
        print(f"Session not found: {session_prefix}", file=sys.stderr)
        return
    for sid, title in rows:
        cur.execute("UPDATE sessions SET excluded = ? WHERE session_id = ?",
                    (1 if exclude else 0, sid))
        action = "excluded" if exclude else "unexcluded"
        print(f"  {action}: {sid[:8]}  {(title or '')[:60]}")
    conn.commit()


def list_excluded(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT session_id, started_at, title FROM sessions WHERE excluded = 1 ORDER BY started_at"
    ).fetchall()
    if not rows:
        print("(no excluded sessions)")
        return
    for sid, started, title in rows:
        print(f"  {sid[:8]}  {(started or '')[:10]}  {(title or '')[:60]}")


def search(conn: sqlite3.Connection, args) -> list[dict]:
    cur = conn.cursor()

    if args.session:
        rows = cur.execute(
            """SELECT m.message_id, m.session_id, m.timestamp, m.role, m.type,
                      m.text_content, m.tool_names, s.title
               FROM messages m
               LEFT JOIN sessions s ON s.session_id = m.session_id
               WHERE m.session_id LIKE ? || '%'
               ORDER BY m.timestamp ASC""",
            (args.session,)
        ).fetchall()
        return [dict(zip(
            ["message_id", "session_id", "timestamp", "role", "type",
             "text_content", "tool_names", "title", "snippet"],
            list(r) + [r[5] or ""]
        )) for r in rows]

    if not args.query:
        print("Either --session or a query string is required.", file=sys.stderr)
        return []

    where = ["messages_fts MATCH ?"]
    params: list = [args.query]

    if args.all_roles:
        pass
    elif getattr(args, "roles", None):
        role_list = [r.strip() for r in args.roles.split(",") if r.strip()]
        if len(role_list) == 1:
            where.append("m.role = ?")
            params.append(role_list[0])
        else:
            placeholders = ",".join("?" * len(role_list))
            where.append(f"m.role IN ({placeholders})")
            params.extend(role_list)
    elif args.role:
        where.append("m.role = ?")
        params.append(args.role)
    else:
        where.append("m.role = 'user'")  # default: signal-only

    if not args.include_excluded:
        where.append("(s.excluded IS NULL OR s.excluded = 0)")

    if args.from_date:
        where.append("m.timestamp >= ?")
        params.append(local_bound_to_utc(args.from_date, end_of_day=False))
    if args.to_date:
        where.append("m.timestamp <= ?")
        params.append(local_bound_to_utc(args.to_date, end_of_day=True))
    if args.has_tool_use:
        where.append("m.has_tool_use = 1")
    if args.source:
        where.append("s.source = ?")
        params.append(args.source)
    elif not args.include_subagents:
        where.append("(s.source IS NULL OR s.source != 'subagent')")

    sql = f"""
        SELECT m.message_id, m.session_id, m.timestamp, m.role, m.type,
               m.text_content, m.tool_names, s.title,
               snippet(messages_fts, 0, '«', '»', '…', 12) AS snippet,
               bm25(messages_fts) AS rank,
               s.source, s.agent_type, s.parent_session_id
        FROM messages_fts
        JOIN messages m ON m.rowid = messages_fts.rowid
        LEFT JOIN sessions s ON s.session_id = m.session_id
        WHERE {" AND ".join(where)}
        ORDER BY rank
        LIMIT ?
    """
    # Overfetch: dedup and the code-content filter both drop rows after the
    # SQL LIMIT, so fetch extra candidates to still return up to args.limit.
    params.append(args.limit * 4)
    rows = cur.execute(sql, params).fetchall()
    results = [dict(zip(
        ["message_id", "session_id", "timestamp", "role", "type",
         "text_content", "tool_names", "title", "snippet", "rank",
         "source", "agent_type", "parent_session_id"],
        r
    )) for r in rows]
    if not getattr(args, "include_code", False):
        results = [r for r in results if not _looks_like_code(r.get("text_content") or "")]
    if not getattr(args, "include_rag_echo", False):
        results = [r for r in results if not _looks_like_rag_echo(r.get("text_content") or "")]
    return _dedup_by_text(results)[:args.limit]


def _score_transcripts(conn, q_vec, args) -> list[dict]:
    """Score all transcript embeddings and return sorted results."""
    rows = conn.execute("""
        SELECT e.message_id, e.vector,
               m.session_id, m.timestamp, m.role, m.text_content,
               s.title, s.excluded, s.source
        FROM embeddings e
        JOIN messages m ON m.message_id = e.message_id
        LEFT JOIN sessions s ON s.session_id = m.session_id
        WHERE length(m.text_content) >= 40
    """).fetchall()

    source_filter = getattr(args, "source", None)
    include_subagents = getattr(args, "include_subagents", False)

    results = []
    for mid, blob, sid, ts, role, text, title, excluded, source in rows:
        if not args.include_excluded and excluded:
            continue
        if source_filter:
            if source != source_filter:
                continue
        elif not include_subagents and source == "subagent":
            continue
        roles_filter = getattr(args, "roles", None)
        if args.all_roles:
            pass
        elif roles_filter:
            role_list = [r.strip() for r in roles_filter.split(",") if r.strip()]
            if role not in role_list:
                continue
        elif getattr(args, "role", None):
            if role != args.role:
                continue
        else:
            if role != "user":
                continue
        if getattr(args, "from_date", None) and ts and ts < local_bound_to_utc(args.from_date, end_of_day=False):
            continue
        if getattr(args, "to_date", None) and ts and ts > local_bound_to_utc(args.to_date, end_of_day=True):
            continue
        if not getattr(args, "include_code", False) and _looks_like_code(text):
            continue
        if not getattr(args, "include_rag_echo", False) and _looks_like_rag_echo(text):
            continue

        vec = _blob_to_vec(blob)
        cosine = _cosine(q_vec, vec)
        score = cosine * ROLE_WEIGHTS.get(role, 0.5) * _time_decay(ts)
        results.append({
            "source": "transcript",
            "message_id": mid,
            "session_id": sid,
            "timestamp": ts,
            "role": role,
            "text_content": text,
            "title": title or "",
            "score": score,
            "cosine": cosine,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _score_file_chunks(conn, q_vec, source_type_filter: str = "all") -> list[dict]:
    """Score file_chunks, optionally filtered by source_type."""
    if source_type_filter == "memory":
        sql = "SELECT id, path, section, text, weight, vector FROM file_chunks WHERE source_type = 'memory'"
    elif source_type_filter == "project":
        sql = "SELECT id, path, section, text, weight, vector FROM file_chunks WHERE source_type != 'memory'"
    else:
        sql = "SELECT id, path, section, text, weight, vector FROM file_chunks"

    rows = conn.execute(sql).fetchall()
    results = []
    for chunk_id, path, section, text, weight, blob in rows:
        vec = _blob_to_vec(blob)
        cosine = _cosine(q_vec, vec)
        results.append({
            "source": "file",
            "chunk_id": chunk_id,
            "path": path,
            "section": section or "",
            "text_content": text,
            "score": cosine * weight,
            "cosine": cosine,
            "timestamp": None,
            "role": "file",
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def search_semantic(conn: sqlite3.Connection, args, config: ProjectConfig) -> list[dict]:
    """Semantic search with per-corpus pooling to prevent diary mass effect.

    With --corpus all (default): merges three pools with guaranteed minimum slots:
      - memory files (rules, patterns, biographical)
      - project files (diary, protocols, medical)
      - transcripts (user messages)

    With --corpus files/transcripts: single-pool search, no pooling.
    """
    model = _load_model(config.model_name)
    q_vec = model.encode([args.query], normalize_embeddings=True)[0]

    corpus = getattr(args, "corpus", "all")

    if corpus == "transcripts":
        results = _score_transcripts(conn, q_vec, args)
        return _dedup_by_text(results)[:args.limit]

    if corpus == "files":
        results = _score_file_chunks(conn, q_vec, "all")
        return _dedup_by_text(results)[:args.limit]

    # corpus == "all" — per-corpus pooling
    pool_t = _score_transcripts(conn, q_vec, args)
    pool_m = _score_file_chunks(conn, q_vec, "memory")
    pool_p = _score_file_chunks(conn, q_vec, "project")

    # Overfetch k so dedup (which can drop pooled slots) still leaves
    # args.limit results after the final trim.
    merged = _merge_pools(
        {"transcript": pool_t, "memory": pool_m, "project": pool_p},
        k=args.limit * 2,
        guarantees=config.pool_guarantees,
    )
    return _dedup_by_text(merged)[:args.limit]


def search_hybrid(conn: sqlite3.Connection, args, config: ProjectConfig) -> list[dict]:
    """FTS retrieval + semantic reranking.

    Falls back to pure semantic search when FTS finds zero lexical candidates
    (essay-phrased/reflective queries routinely share no word-stem with the
    indexed text) — otherwise --hybrid silently returns nothing even when
    semantically relevant content exists, contradicting its own docstring
    promise of "best thematic recall"."""
    orig_limit = args.limit
    args.limit = orig_limit * 3
    fts_results = search(conn, args)
    args.limit = orig_limit

    if not fts_results:
        return search_semantic(conn, args, config)

    try:
        model = _load_model(config.model_name)
    except ModuleNotFoundError:
        # Model unavailable even after _ensure_semantic_capable's re-exec
        # attempt (e.g. fallback interpreter missing on this machine) — return
        # the FTS candidates unranked rather than losing them entirely.
        print("[hybrid] semantic rerank unavailable (sentence_transformers not "
              "importable) — showing unranked FTS results", file=sys.stderr)
        return fts_results[:orig_limit]

    q_vec = model.encode([args.query], normalize_embeddings=True)[0]

    reranked = []
    for r in fts_results:
        mid = r["message_id"]
        row = conn.execute(
            "SELECT vector FROM embeddings WHERE message_id = ?", (mid,)
        ).fetchone()
        if not row:
            reranked.append({**r, "score": 0.0, "cosine": 0.0})
            continue

        vec = _blob_to_vec(row[0])
        cosine = _cosine(q_vec, vec)
        role_w = ROLE_WEIGHTS.get(r.get("role", "user"), 0.5)
        decay = _time_decay(r.get("timestamp"))
        reranked.append({**r, "score": cosine * role_w * decay, "cosine": cosine})

    reranked.sort(key=lambda r: r["score"], reverse=True)
    return _dedup_by_text(reranked)[:orig_limit]


def search_artifacts(conn: sqlite3.Connection, query: str, args) -> list[dict]:
    """Search the artifacts (Write/Edit file operations) index."""
    cur = conn.cursor()
    params: list = [query]
    where_extra = ""
    if getattr(args, "from_date", None):
        where_extra += " AND a.timestamp >= ?"
        params.append(local_bound_to_utc(args.from_date, end_of_day=False))
    if getattr(args, "to_date", None):
        where_extra += " AND a.timestamp <= ?"
        params.append(local_bound_to_utc(args.to_date, end_of_day=True))
    params.append(args.limit)
    sql = f"""
        SELECT a.id, a.session_id, a.timestamp, a.tool_name, a.file_path, a.content_preview,
               snippet(artifacts_fts, 0, '«', '»', '…', 10) AS fp_snip,
               snippet(artifacts_fts, 1, '«', '»', '…', 12) AS ct_snip
        FROM artifacts_fts
        JOIN artifacts a ON a.id = artifacts_fts.rowid
        LEFT JOIN sessions s ON s.session_id = a.session_id
        WHERE artifacts_fts MATCH ?{where_extra}
        ORDER BY bm25(artifacts_fts)
        LIMIT ?
    """
    rows = cur.execute(sql, params).fetchall()
    return [dict(zip(
        ["id", "session_id", "timestamp", "tool_name", "file_path",
         "content_preview", "fp_snip", "ct_snip"],
        r
    )) for r in rows]


def render_artifacts(results: list[dict], full: bool = False) -> str:
    if not results:
        return "(no artifact results)"
    parts = []
    for i, r in enumerate(results, 1):
        ts = fmt_ts(r.get("timestamp", ""))
        sid = short(r.get("session_id", ""))
        fp = r.get("file_path", "")
        snip = r.get("ct_snip") or r.get("content_preview") or ""
        parts.append(f"{i:3}. [{ts}] {sid}  {r['tool_name']}  {fp}")
        if snip and not full:
            parts.append(f"     {snip[:200]}")
        elif full and r.get("content_preview"):
            for line in (r["content_preview"] or "").splitlines():
                parts.append(f"     {line}")
        parts.append("")
    return "\n".join(parts)


def preview_sessions(conn: sqlite3.Connection, results: list[dict], n: int) -> str:
    """Show first N user messages from each unique session found in results."""
    seen_sids: list[str] = []
    for r in results:
        sid = r.get("session_id")
        if sid and sid not in seen_sids:
            seen_sids.append(sid)

    parts = []
    for sid in seen_sids:
        all_rows = conn.execute(
            """SELECT m.timestamp, m.text_content
               FROM messages m
               WHERE m.session_id = ? AND m.role = 'user' AND m.text_content IS NOT NULL
               ORDER BY m.timestamp
               LIMIT ?""",
            (sid, n * 3),
        ).fetchall()
        _SKIP_PREFIXES = ("<local-command", "<command-", "<ide_selection", "<ide_opened_file",
                          "<system-reminder", "<user-prompt-submit-hook")
        rows = []
        for ts, text in all_rows:
            t = (text or "").strip()
            if len(t) < 15:
                continue
            if any(t.startswith(p) for p in _SKIP_PREFIXES):
                continue
            tag_chars = len(re.findall(r'<[^>]+>', t))
            if tag_chars > len(t) * 0.8:
                continue
            rows.append((ts, t))
            if len(rows) >= n:
                break

        title_row = conn.execute(
            "SELECT title, started_at FROM sessions WHERE session_id = ?", (sid,)
        ).fetchone()
        title = (title_row[0] or "") if title_row else ""
        started = fmt_ts(title_row[1] or "") if title_row else ""
        parts.append(f"── {sid[:8]}  {started}  {title[:60]} ──")
        for ts, text in rows:
            parts.append(f"  [{fmt_ts(ts)}] {text[:200]}")
        parts.append("")
    return "\n".join(parts) if parts else "(no sessions found)"


def render_text(results: list[dict], full: bool, width: int = 100) -> str:
    if not results:
        return "(no results)"
    parts = []
    seen_bodies: set[str] = set()
    shown = 0
    for r in results:
        body = r.get("snippet") if (not full and "snippet" in r) else r.get("text_content")
        body = (body or "").strip()
        # Dedup on the text actually printed, not the raw message: two
        # different sessions can share an identical snippet/passage (e.g. a
        # file re-read unchanged, or a quote from an evolving draft) even
        # when their full underlying messages differ elsewhere.
        body_key = re.sub(r"\s+", " ", body).lower()[:400]
        if body_key and body_key in seen_bodies:
            continue
        if body_key:
            seen_bodies.add(body_key)

        shown += 1
        source = r.get("source", "transcript")
        score_str = f"  score={r['score']:.3f}" if "score" in r else ""

        if source == "file":
            section = r.get("section") or ""
            section_part = f" § {section[:60]}" if section else ""
            parts.append(f"{shown:3}. [файл: {r['path']}{section_part}]{score_str}")
        else:
            ts = fmt_ts(r.get("timestamp", ""))
            sid = short(r.get("session_id", ""))
            role = r.get("role", "")
            title = (r.get("title") or "").strip()
            title_part = f" · {title[:60]}" if title else ""
            agent_part = ""
            if r.get("source") == "subagent":
                parent = short(r.get("parent_session_id") or "")
                agent_part = f"  [agent:{r.get('agent_type') or '?'} ← {parent}]"
            parts.append(f"{shown:3}. [{ts}] {sid} {role}{title_part}{agent_part}{score_str}")

        if not full:
            parts.append(textwrap.indent(textwrap.fill(body[:400], width=width), "     "))
        else:
            for line in body.splitlines():
                parts.append(textwrap.indent(line, "     "))
        parts.append("")
    if shown == 0:
        return "(no results)"
    return "\n".join(parts)


def main(config: ProjectConfig, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?", help="FTS5 query string")
    parser.add_argument("--db", type=Path, default=config.db_path)
    parser.add_argument("--session", help="Retrieve full session by id-prefix")
    parser.add_argument("--role", choices=["user", "assistant", "tool_result", "attachment", "system"],
                        help="Filter by single role (overrides default user-only)")
    parser.add_argument("--roles", metavar="ROLE[,ROLE]",
                        help="Comma-separated roles, e.g. user,assistant (overrides default user-only)")
    parser.add_argument("--all-roles", action="store_true",
                        help="Search all roles (default is user only)")
    parser.add_argument("--exclude-session", metavar="PREFIX",
                        help="Mark session(s) as excluded from default searches")
    parser.add_argument("--unexclude-session", metavar="PREFIX",
                        help="Remove excluded flag from session(s)")
    parser.add_argument("--list-excluded", action="store_true",
                        help="List all excluded sessions")
    parser.add_argument("--include-excluded", action="store_true",
                        help="Include excluded sessions in search results")
    parser.add_argument("--include-code", action="store_true",
                        help="Include messages that look like source code (excluded by default)")
    parser.add_argument("--include-rag-echo", action="store_true",
                        help="Include this tool's own query.py/restore.py invocations and output "
                             "banners in results (excluded by default)")
    parser.add_argument("--from", dest="from_date", metavar="YYYY-MM-DD",
                        help="Only messages on or after this date")
    parser.add_argument("--to", dest="to_date", metavar="YYYY-MM-DD",
                        help="Only messages on or before this date")
    parser.add_argument("--until", dest="to_date", metavar="YYYY-MM-DD",
                        help="Alias for --to")
    parser.add_argument("--recent", action="store_true",
                        help="Limit to last 30 days (auto sets --from)")
    parser.add_argument("--has-tool-use", action="store_true",
                        help="Only assistant messages that called tools")
    parser.add_argument("--source", choices=["claude", "copilot", "subagent"],
                        help="Filter by source (claude, copilot, or subagent)")
    parser.add_argument("--include-subagents", action="store_true",
                        help="Include subagent transcripts (Task/Agent calls) alongside normal results. "
                             "Excluded by default — noisy, requires 'ingest.py --include-subagents' first.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--full", action="store_true",
                        help="Print full message text instead of FTS snippet")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--semantic", action="store_true",
                        help="Semantic vector search (transcripts + files)")
    parser.add_argument("--hybrid", action="store_true",
                        help="FTS retrieval + semantic reranking (recommended for thematic queries)")
    parser.add_argument("--corpus", choices=["all", "transcripts", "files"], default="all",
                        help="Corpus for semantic/hybrid mode (default: all)")
    parser.add_argument("--no-expand", action="store_true",
                        help="Disable auto-expansion of Cyrillic words (adds * by default)")
    parser.add_argument("--artifacts", metavar="QUERY",
                        help="Search Write/Edit file operations index instead of messages")
    parser.add_argument("--preview", type=int, metavar="N",
                        help="Show first N user messages from each matching session")
    parser.add_argument("--archives", metavar="NAME[,NAME]",
                        help=f"Search multiple archives: {', '.join(KNOWN_ARCHIVES)} or absolute paths")
    args = parser.parse_args(argv)

    if args.recent and not args.from_date:
        args.from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    if args.semantic or args.hybrid:
        _ensure_semantic_capable()

    if getattr(args, "archives", None):
        return _run_multi_archive(args, config)

    _auto_update_if_stale(config, args.db)

    if not args.db.exists():
        print(f"DB not found: {args.db}\nRun ingest.py first.", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    try:
        if args.list_excluded:
            list_excluded(conn)
            return 0
        if args.exclude_session:
            manage_exclusion(conn, args.exclude_session, exclude=True)
            return 0
        if args.unexclude_session:
            manage_exclusion(conn, args.unexclude_session, exclude=False)
            return 0

        if getattr(args, "artifacts", None):
            art_results = search_artifacts(conn, args.artifacts, args)
            if args.json:
                print(json.dumps(art_results, ensure_ascii=False, indent=2))
            else:
                print(render_artifacts(art_results, full=args.full))
            return 0

        if args.query and not args.semantic and not getattr(args, "no_expand", False):
            args.query = _sanitize_fts_query(args.query)
            expanded = _expand_cyrillic(args.query)
            if expanded != args.query:
                print(f"[expand] {expanded}", file=sys.stderr)
                args.query = expanded

        if args.semantic:
            results = search_semantic(conn, args, config)
        elif args.hybrid:
            results = search_hybrid(conn, args, config)
        else:
            results = search(conn, args)

        if getattr(args, "preview", None) and results:
            print(preview_sessions(conn, results, args.preview))
            return 0

    finally:
        conn.close()

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(render_text(results, full=args.full))
    return 0


def _run_multi_archive(args, config: ProjectConfig) -> int:
    """Run search across multiple named or path-specified archives and print results."""
    archive_specs = [s.strip() for s in args.archives.split(",") if s.strip()]
    db_paths: list[tuple[str, Path]] = []
    for spec in archive_specs:
        if spec in KNOWN_ARCHIVES:
            db_paths.append((spec, KNOWN_ARCHIVES[spec]))
        else:
            p = Path(spec)
            db_paths.append((p.parent.parent.name, p))

    artifact_query = getattr(args, "artifacts", None)

    for label, db_path in db_paths:
        print(f"\n{'═' * 60}")
        print(f"  Archive: {label}  ({db_path})")
        print(f"{'═' * 60}")

        if not db_path.exists():
            print(f"  (not found — skipping)")
            continue

        conn = sqlite3.connect(db_path)
        try:
            if artifact_query:
                results = search_artifacts(conn, artifact_query, args)
                print(render_artifacts(results, full=args.full))
            elif args.query and not args.semantic and not getattr(args, "no_expand", False):
                orig_query = args.query
                args.query = _sanitize_fts_query(args.query)
                expanded = _expand_cyrillic(args.query)
                if expanded != args.query:
                    print(f"[expand] {expanded}", file=sys.stderr)
                    args.query = expanded
                if args.semantic:
                    results = search_semantic(conn, args, config)
                elif args.hybrid:
                    results = search_hybrid(conn, args, config)
                else:
                    results = search(conn, args)
                args.query = orig_query
                print(render_text(results, full=args.full))
            elif args.semantic:
                results = search_semantic(conn, args, config)
                print(render_text(results, full=args.full))
            elif args.hybrid:
                results = search_hybrid(conn, args, config)
                print(render_text(results, full=args.full))
            else:
                results = search(conn, args)
                print(render_text(results, full=args.full))
        finally:
            conn.close()
    return 0
