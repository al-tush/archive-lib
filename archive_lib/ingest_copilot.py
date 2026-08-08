#!/usr/bin/env python3
"""
Ingest GitHub Copilot chat history (chatSessions JSONL) into the shared transcripts.db.

Source: ~/Library/Application Support/Code/User/workspaceStorage/<config.copilot_ws_id>/chatSessions/
Target: config.db_path, with source='copilot'.

The chatSessions JSONL format is a delta-patch log:
  kind=0  — initial state snapshot  (v = full session state dict)
  kind=1  — partial patch            (k = path list, v = new value; merge/update)
  kind=2  — replace patch            (k = path list, v = new value; full replace)

Relevant patches:
  k=['requests']          → full requests array snapshot; each entry has message.text (user turn)
  k=['requests', N, 'response'], kind=2 → AI response items array for turn N
  k=['customTitle']       → session title set by user or auto-generated

Response items array: list of dicts. Text chunks are items WITHOUT a 'kind' field
(they have: value, supportThemeIcons, supportHtml, baseUri). Items WITH a 'kind' field
are: 'thinking' (skipped, same policy as Claude Code), 'toolInvocationSerialized',
'codeblockUri', 'textEditGroup', 'undoStop' — all skipped for text_content.

Stdlib-only: no external dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import ProjectConfig
from .hybrid_words import normalize_hybrid_words
from .ingest import date_from_iso, short_uuid
from .schema import open_db


# --- Parsing ---------------------------------------------------------------

def ms_epoch_to_iso(ms: int | float | None) -> str:
    """Convert millisecond epoch to ISO-8601 string."""
    if not ms:
        return ""
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )
    except (ValueError, OSError):
        return ""


def extract_response_text(response_items: list) -> str:
    """
    Extract plain text from a Copilot response items array.

    Text chunks are dicts WITHOUT a 'kind' field — they have 'value', 'supportThemeIcons', etc.
    Items WITH 'kind' (thinking, toolInvocationSerialized, codeblockUri, etc.) are skipped.
    This mirrors Claude Code's policy of skipping thinking blocks.
    """
    parts: list[str] = []
    for item in response_items:
        if not isinstance(item, dict):
            continue
        if "kind" in item:
            continue
        val = item.get("value")
        if isinstance(val, str) and val.strip():
            parts.append(val)
    return "\n\n".join(parts).strip()


def parse_chatsession(jsonl_path: Path) -> dict | None:
    """
    Parse a chatSessions JSONL file into a structured session dict.

    Returns:
        {
          'session_id': str,
          'title': str | None,
          'created_at': str,   # ISO from first request timestamp or creationDate
          'ended_at': str,     # ISO from last request timestamp
          'turns': [
            {
              'index': int,
              'request_id': str,
              'response_id': str,
              'timestamp': str,      # ISO
              'model_id': str,
              'user_text': str,
              'ai_text': str,
            }, ...
          ]
        }

    Returns None only when the file itself can't be read (turns=[] is a valid
    result for a chat panel that was opened but never used).
    """
    session_id = jsonl_path.stem
    title: str | None = None
    creation_date_ms: int | None = None

    requests_ordered: list[str] = []
    requests_by_id: dict[str, dict] = {}
    response_map: dict[int, list] = {}

    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        print(f"  [WARN] Cannot read {jsonl_path.name}: {e}", file=sys.stderr)
        return None

    for lineno, raw_line in enumerate(lines, 1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError as e:
            print(f"  [WARN] {jsonl_path.name}:{lineno} JSON decode error: {e}", file=sys.stderr)
            continue

        event_kind = obj.get("kind")
        k = obj.get("k", [])
        v = obj.get("v")

        if event_kind == 0:
            state = v if isinstance(v, dict) else {}
            creation_date_ms = state.get("creationDate") or creation_date_ms
            if state.get("customTitle"):
                title = state["customTitle"]
            for r in state.get("requests", []):
                if isinstance(r, dict):
                    _add_request(requests_ordered, requests_by_id, r)
            continue

        if not isinstance(k, list):
            continue

        if k == ["customTitle"] and isinstance(v, str) and v.strip():
            title = v.strip()
            continue

        if k == ["requests"] and isinstance(v, list):
            for r in v:
                if isinstance(r, dict):
                    _add_request(requests_ordered, requests_by_id, r)
            continue

        if (
            event_kind == 2
            and len(k) == 3
            and k[2] == "response"
            and isinstance(k[1], int)
            and isinstance(v, list)
        ):
            response_map[k[1]] = v
            continue

    turns: list[dict] = []
    for n, rid in enumerate(requests_ordered):
        req = requests_by_id[rid]
        resp_v = response_map.get(n, [])
        user_text = req.get("user_text", "").strip()
        ai_text = extract_response_text(resp_v)

        if not user_text and not ai_text:
            continue

        turns.append(
            {
                "index": n,
                "request_id": req.get("requestId") or f"{session_id}_req_{n}",
                "response_id": req.get("responseId") or f"{session_id}_resp_{n}",
                "timestamp": ms_epoch_to_iso(req.get("timestamp") or creation_date_ms),
                "model_id": req.get("modelId") or "",
                "user_text": user_text,
                "ai_text": ai_text,
            }
        )

    # turns may legitimately be empty (chat panel opened, nothing ever sent).
    # Still return a session dict (with 0 turns) rather than None, so the caller
    # records it in the DB — otherwise these files never become "known" and
    # every query.py run re-flags them as new, triggering auto-update forever.
    timestamps = [t["timestamp"] for t in turns if t["timestamp"]]
    started_at = min(timestamps) if timestamps else ms_epoch_to_iso(creation_date_ms)
    ended_at = max(timestamps) if timestamps else started_at

    return {
        "session_id": session_id,
        "title": title,
        "created_at": started_at,
        "ended_at": ended_at,
        "turns": turns,
    }


def _add_request(ordered: list, by_id: dict, r: dict) -> None:
    """Add or update a request in the deduplicated ordered registry."""
    rid = r.get("requestId")
    if not rid:
        return
    if rid not in by_id:
        by_id[rid] = {}
        ordered.append(rid)
    entry = by_id[rid]
    msg = r.get("message", {})
    if isinstance(msg, dict) and msg.get("text"):
        entry["user_text"] = msg["text"]
    for field in ("requestId", "responseId", "timestamp", "modelId"):
        if r.get(field) and not entry.get(field):
            entry[field] = r[field]


# --- DB insertion ----------------------------------------------------------

def insert_session(session: dict, conn, dry_run: bool = False,
                   source_mtime: int = 0, source_size: int = 0) -> dict:
    """
    Insert a parsed Copilot session into the DB.
    Each turn produces up to two rows: one user message and one assistant message.
    Returns stats dict.
    """
    session_id = session["session_id"]
    turns = session["turns"]

    messages = []
    for turn in turns:
        ts = turn["timestamp"]
        req_id = turn["request_id"]
        resp_id = turn["response_id"]

        if turn["user_text"]:
            messages.append(
                {
                    "message_id": req_id,
                    "session_id": session_id,
                    "parent_id": None,
                    "timestamp": ts,
                    "role": "user",
                    "type": "user",
                    "text_content": normalize_hybrid_words(
                        turn["user_text"], warn_context=f"{session_id[:8]}:{req_id[:8]}"
                    ),
                    "has_tool_use": 0,
                    "tool_names": None,
                    "raw_json": json.dumps(
                        {"requestId": req_id, "modelId": turn["model_id"]},
                        ensure_ascii=False,
                    ),
                }
            )

        if turn["ai_text"]:
            messages.append(
                {
                    "message_id": resp_id,
                    "session_id": session_id,
                    "parent_id": req_id if turn["user_text"] else None,
                    "timestamp": ts,
                    "role": "assistant",
                    "type": "assistant",
                    "text_content": normalize_hybrid_words(
                        turn["ai_text"], warn_context=f"{session_id[:8]}:{resp_id[:8]}"
                    ),
                    "has_tool_use": 0,
                    "tool_names": None,
                    "raw_json": json.dumps(
                        {"responseId": resp_id, "modelId": turn["model_id"]},
                        ensure_ascii=False,
                    ),
                }
            )

    if dry_run:
        return {
            "session_id": session_id,
            "messages": len(messages),
            "title": session.get("title"),
            "started": session["created_at"],
            "ended": session["ended_at"],
        }

    cur = conn.cursor()
    cur.execute("DELETE FROM attachments WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    cur.execute(
        """INSERT INTO sessions
           (session_id, started_at, ended_at, title, cwd, entrypoint, version,
            git_branch, message_count, source, source_mtime, source_size)
           VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, 'copilot', ?, ?)""",
        (
            session_id,
            session["created_at"],
            session["ended_at"],
            session.get("title"),
            len(messages),
            source_mtime,
            source_size,
        ),
    )

    cur.executemany(
        """INSERT INTO messages
           (message_id, session_id, parent_id, timestamp, role, type,
            text_content, has_tool_use, tool_names, raw_json)
           VALUES (:message_id, :session_id, :parent_id, :timestamp, :role, :type,
                   :text_content, :has_tool_use, :tool_names, :raw_json)""",
        messages,
    )

    conn.commit()

    return {
        "session_id": session_id,
        "messages": len(messages),
        "title": session.get("title"),
        "started": session["created_at"],
        "ended": session["ended_at"],
    }


# --- JSONL reconstruction ---------------------------------------------------
#
# ingest.py copies Claude Code JSONL (and subagent JSONL) into archive/transcripts/
# as a durable, git-tracked backup — the live source directories get pruned by
# Claude Code over time, so that copy is the only long-term record. Copilot
# sessions never got this treatment: chatSessions live only in VS Code's
# workspaceStorage (not git-tracked, not backed up), and only their *content*
# (not the original delta-patch JSONL) was ever persisted, in transcripts.db.
# This reconstructs a Claude-Code-shaped JSONL (not byte-identical to the
# original chatSession format, but carries the same user/assistant text) so
# Copilot sessions get the same durable git-tracked backup as everything else.

def build_jsonl_events(session: dict) -> list[dict]:
    """Reconstruct Claude-Code-shaped JSONL events from a parsed Copilot session."""
    events: list[dict] = []
    session_id = session["session_id"]
    for turn in session["turns"]:
        ts = turn["timestamp"]
        if turn["user_text"]:
            events.append({
                "type": "user",
                "uuid": turn["request_id"],
                "sessionId": session_id,
                "timestamp": ts,
                "message": {"role": "user", "content": [{"type": "text", "text": turn["user_text"]}]},
            })
        if turn["ai_text"]:
            events.append({
                "type": "assistant",
                "uuid": turn["response_id"],
                "parentUuid": turn["request_id"] if turn["user_text"] else None,
                "sessionId": session_id,
                "timestamp": ts,
                "message": {
                    "role": "assistant",
                    "model": turn["model_id"] or "copilot",
                    "content": [{"type": "text", "text": turn["ai_text"]}],
                },
            })
    return events


def write_session_jsonl(session: dict, copilot_dir: Path) -> Path | None:
    """Write a reconstructed JSONL for this session if one doesn't already exist
    (matched by short-hash, same convention as ingest.py's archive copies).
    Returns the written path, or None if a matching file already existed."""
    session_id = session["session_id"]
    short = short_uuid(session_id)
    if copilot_dir.is_dir() and any(short in p.stem for p in copilot_dir.glob("*.jsonl")):
        return None

    events = build_jsonl_events(session)
    if not events:
        return None

    copilot_dir.mkdir(parents=True, exist_ok=True)
    date = date_from_iso(session.get("created_at") or "")
    name = f"{date}-{short}.jsonl" if date else f"{session_id}.jsonl"
    dest = copilot_dir / name
    with dest.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return dest


# --- Main ------------------------------------------------------------------

def main(config: ProjectConfig, argv: list[str]) -> int:
    default_chatsessions = config.copilot_chatsessions_dir
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-dir", type=Path, default=default_chatsessions,
        help=f"chatSessions directory (default: {default_chatsessions})"
    )
    parser.add_argument(
        "--db", type=Path, default=config.db_path,
        help=f"Target SQLite DB (default: {config.db_path})"
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse without writing to DB")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files (0 = all)")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest all files, ignoring cached mtime/size")
    args = parser.parse_args(argv)

    source_dir: Path | None = args.source_dir
    if source_dir is None:
        print("No Copilot workspace configured for this project — skipping.", file=sys.stderr)
        return 0
    if not source_dir.is_dir():
        print(f"chatSessions directory not found: {source_dir}", file=sys.stderr)
        return 1

    files = sorted(source_dir.glob("*.jsonl"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("No chatSessions JSONL files found.")
        return 0

    if not args.quiet:
        print(f"{'DRY RUN: ' if args.dry_run else ''}Processing {len(files)} Copilot sessions from:")
        print(f"  {source_dir}")
        print()

    conn = None if args.dry_run else open_db(args.db)
    copilot_jsonl_dir = config.transcripts_dir / "copilot"

    total_sessions = 0
    total_messages = 0
    total_exported = 0
    skipped_unchanged = 0
    skipped_unreadable = 0

    for path in files:
        stat = path.stat()
        short = short_uuid(path.stem)
        jsonl_exists = copilot_jsonl_dir.is_dir() and any(
            short in p.stem for p in copilot_jsonl_dir.glob("*.jsonl")
        )
        if not args.dry_run and not args.force:
            row = conn.execute(
                "SELECT source_mtime, source_size, message_count FROM sessions WHERE session_id = ?",
                (path.stem,)
            ).fetchone()
            if row and row[0] is not None and row[0] == stat.st_mtime_ns and row[1] == stat.st_size:
                # A tombstoned empty session (message_count=0) has no jsonl to
                # verify against — mtime/size match is enough to call it unchanged.
                if jsonl_exists or row[2] == 0:
                    skipped_unchanged += 1
                    if not args.quiet:
                        print(f"  {path.name[:42]:42}  (unchanged — skipped)")
                    continue

        session = parse_chatsession(path)
        if session is None:
            skipped_unreadable += 1
            if not args.quiet:
                print(f"  {path.name[:42]:42}  (unreadable — skipped)")
            continue

        stats = insert_session(session, conn, dry_run=args.dry_run,
                               source_mtime=stat.st_mtime_ns, source_size=stat.st_size)
        total_sessions += 1
        total_messages += stats["messages"]

        exported_path = None
        if not args.dry_run:
            exported_path = write_session_jsonl(session, copilot_jsonl_dir)
            if exported_path:
                total_exported += 1

        if not args.quiet:
            title = (stats.get("title") or "")[:50]
            started = (stats.get("started") or "")[:10]
            export_str = "  [jsonl exported]" if exported_path else ""
            print(
                f"  {path.name[:42]:42}  msgs={stats['messages']:4}  "
                f"started={started}  {title}{export_str}"
            )

    if not args.quiet:
        skip_parts = []
        if skipped_unchanged:
            skip_parts.append(f"{skipped_unchanged} unchanged")
        if skipped_unreadable:
            skip_parts.append(f"{skipped_unreadable} unreadable")
        skip_str = f", {', '.join(skip_parts)} skipped" if skip_parts else ""
        export_str = f", {total_exported} JSONL exported to {copilot_jsonl_dir}" if total_exported else ""
        print(f"\nTotal: {total_sessions} sessions ingested, {total_messages} messages{skip_str}{export_str}")
        if conn is not None:
            print(f"DB: {args.db}")

    if conn is not None:
        conn.close()
    return 0
