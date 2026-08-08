#!/usr/bin/env python3
"""
Ingest Claude Code session transcripts (JSONL) into a SQLite archive with FTS5 search.

Source: <config.claude_source_dir>/*.jsonl
Target: <config.db_path>
Side effect: copies JSONL files to <config.transcripts_dir> with date prefix.

The script is idempotent — re-running on the same JSONL replaces the session/messages.

Stdlib-only: no external dependencies.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import shutil
import sqlite3
import sys
import time
from pathlib import Path

from .config import ProjectConfig
from .hybrid_words import normalize_hybrid_words
from .schema import open_db
from .timeutil import LOCAL_TZ

# Types we care about for archive content.
RELEVANT_TYPES = {"user", "assistant", "tool_result", "ai-title", "attachment", "system"}
# Types we explicitly skip (internal Claude Code state).
SKIP_TYPES = {"queue-operation", "last-prompt", "file-history-snapshot", "create", "update", "text"}

# Common media types — extension fallback overrides for clarity (.jpg vs .jpe, etc.).
MEDIA_TYPE_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
}


# --- Helpers ---------------------------------------------------------------

def extract_text_and_tools(content) -> tuple[str, list[str]]:
    """From message.content (list of blocks), return (text_aggregate, tool_names)."""
    texts: list[str] = []
    tool_names: list[str] = []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return "", []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            t = block.get("text")
            if t:
                texts.append(t)
        elif btype == "tool_use":
            name = block.get("name")
            if name:
                tool_names.append(name)
        elif btype == "tool_result":
            r = block.get("content")
            if isinstance(r, str):
                texts.append(r)
            elif isinstance(r, list):
                for sub in r:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        st = sub.get("text")
                        if st:
                            texts.append(st)
        # thinking blocks are intentionally not aggregated into text_content
    return "\n\n".join(texts), tool_names


def parse_message_event(event: dict) -> dict | None:
    """Convert a JSONL event into a message row dict, or None if not relevant."""
    etype = event.get("type")
    if etype not in RELEVANT_TYPES:
        return None
    uuid = event.get("uuid")
    if not uuid:
        return None
    timestamp = event.get("timestamp") or ""
    parent = event.get("parentUuid")
    session_id = event.get("sessionId") or ""

    role = ""
    text = ""
    tool_names: list[str] = []
    has_tool_use = 0

    if etype == "user":
        role = "user"
        msg = event.get("message") or {}
        text, tool_names = extract_text_and_tools(msg.get("content"))
    elif etype == "assistant":
        role = "assistant"
        msg = event.get("message") or {}
        text, tool_names = extract_text_and_tools(msg.get("content"))
        has_tool_use = 1 if tool_names else 0
    elif etype == "tool_result":
        role = "tool_result"
        msg = event.get("message") or {}
        if msg:
            text, tool_names = extract_text_and_tools(msg.get("content"))
        if not text:
            r = event.get("toolUseResult")
            if isinstance(r, str):
                text = r
            elif isinstance(r, dict):
                t = r.get("content") or r.get("text") or r.get("stdout")
                if isinstance(t, str):
                    text = t
    elif etype == "attachment":
        role = "attachment"
        att = event.get("attachment") or {}
        stdout = att.get("stdout")
        if isinstance(stdout, str) and stdout:
            text = stdout
        else:
            content = att.get("content") or ""
            if isinstance(content, str):
                text = content
    elif etype == "ai-title":
        role = "ai-title"
        text = event.get("title") or ""
    elif etype == "system":
        role = "system"
        msg = event.get("message") or {}
        text, _ = extract_text_and_tools(msg.get("content"))
        if not text:
            t = event.get("text") or event.get("content")
            if isinstance(t, str):
                text = t

    text = normalize_hybrid_words(text, warn_context=f"{session_id[:8]}:{uuid[:8]}")

    return {
        "message_id": uuid,
        "session_id": session_id,
        "parent_id": parent,
        "timestamp": timestamp,
        "role": role,
        "type": etype,
        "text_content": text,
        "has_tool_use": has_tool_use,
        "tool_names": ",".join(tool_names) if tool_names else None,
    }


def date_from_iso(ts: str) -> str:
    """Return YYYY-MM-DD from ISO timestamp (local tz), or '' if unparseable."""
    if not ts:
        return ""
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d")
    except ValueError:
        return ts[:10] if len(ts) >= 10 else ""


def yearmonth_from_iso(ts: str) -> str:
    """Return YYYY-MM from ISO timestamp (local tz)."""
    if not ts:
        return "unknown"
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m")
    except ValueError:
        return ts[:7] if len(ts) >= 7 else "unknown"


def short_uuid(session_id: str) -> str:
    return session_id.split("-")[0][:8] if session_id else "unknown"


def ext_for_media_type(media_type: str | None) -> str:
    if not media_type:
        return ".bin"
    if media_type in MEDIA_TYPE_EXT:
        return MEDIA_TYPE_EXT[media_type]
    guess = mimetypes.guess_extension(media_type)
    return guess or ".bin"


def collect_attachments_from_content(content, message_id: str, session_id: str,
                                     timestamp: str) -> list[dict]:
    """Find image/document blocks in message.content and return attachment records (without files written yet)."""
    out: list[dict] = []
    if not isinstance(content, list):
        return out
    for pos, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype not in {"image", "document"}:
            continue
        source = block.get("source") or {}
        stype = source.get("type")
        media_type = source.get("media_type")
        rec = {
            "message_id": message_id,
            "session_id": session_id,
            "position": pos,
            "kind": btype,
            "media_type": media_type,
            "source_type": stype,
            "original_path": None,
            "archive_path": None,
            "sha256": None,
            "size_bytes": None,
            "extracted": 0,
            "error": None,
            "_data": source.get("data"),
            "_url": source.get("url"),
            "_timestamp": timestamp,
        }
        out.append(rec)
    return out


def collect_path_attachment_from_event(event: dict) -> dict | None:
    """For attachment events that reference a file by path, return a record (no file copied yet)."""
    if event.get("type") != "attachment":
        return None
    att = event.get("attachment") or {}
    path = att.get("path") or att.get("filePath") or att.get("file")
    if not path:
        return None
    msg_id = event.get("uuid") or ""
    session_id = event.get("sessionId") or ""
    ts = event.get("timestamp") or ""
    return {
        "message_id": msg_id,
        "session_id": session_id,
        "position": 0,
        "kind": "file",
        "media_type": att.get("mediaType") or att.get("type"),
        "source_type": "path",
        "original_path": str(path),
        "archive_path": None,
        "sha256": None,
        "size_bytes": None,
        "extracted": 0,
        "error": None,
        "_data": None,
        "_url": None,
        "_timestamp": ts,
    }


def write_attachment_file(rec: dict, files_dir: Path, project_root: Path) -> None:
    """Realize the attachment to disk. Mutates rec to fill in archive_path, sha256, size_bytes, extracted, error."""
    yearmonth = yearmonth_from_iso(rec.get("_timestamp") or "")
    session_short = short_uuid(rec["session_id"])
    msg_short = short_uuid(rec["message_id"])
    pos = rec.get("position") or 0
    target_dir = files_dir / yearmonth / session_short
    target_dir.mkdir(parents=True, exist_ok=True)

    if rec["source_type"] == "base64":
        b64 = rec.get("_data") or ""
        try:
            data = base64.b64decode(b64, validate=False)
        except Exception as e:
            rec["error"] = f"base64 decode: {e}"
            return
        ext = ext_for_media_type(rec.get("media_type"))
        target = target_dir / f"{msg_short}-{pos}{ext}"
        if not target.exists() or target.stat().st_size != len(data):
            target.write_bytes(data)
        rec["archive_path"] = str(target.relative_to(project_root))
        rec["sha256"] = hashlib.sha256(data).hexdigest()
        rec["size_bytes"] = len(data)
        rec["extracted"] = 1
    elif rec["source_type"] == "path":
        src = Path(rec["original_path"])
        if not src.exists():
            rec["error"] = "source path does not exist"
            return
        try:
            data = src.read_bytes()
        except Exception as e:
            rec["error"] = f"read source: {e}"
            return
        ext = src.suffix or ext_for_media_type(rec.get("media_type"))
        target = target_dir / f"{msg_short}-{pos}-{src.name}"
        if target.suffix != ext:
            target = target.with_suffix(ext)
        if not target.exists() or target.stat().st_size != len(data):
            target.write_bytes(data)
        rec["archive_path"] = str(target.relative_to(project_root))
        rec["sha256"] = hashlib.sha256(data).hexdigest()
        rec["size_bytes"] = len(data)
        rec["extracted"] = 1
    elif rec["source_type"] == "url":
        rec["error"] = "url source — not downloaded"
    else:
        rec["error"] = f"unknown source_type: {rec['source_type']}"


# --- Artifact extraction ---------------------------------------------------

def extract_write_edit_artifacts(content) -> list[tuple]:
    """Return [(tool_name, file_path, content_preview)] for Write/Edit tool_use blocks."""
    if not isinstance(content, list):
        return []
    result = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        name = block.get("name")
        if name not in ("Write", "Edit"):
            continue
        inp = block.get("input") or {}
        fp = inp.get("file_path") or inp.get("path") or ""
        if not fp:
            continue
        raw = inp.get("content") or inp.get("new_string") or ""
        preview = str(raw)[:500]
        result.append((name, str(fp), preview))
    return result


def rebuild_artifacts(conn: sqlite3.Connection, quiet: bool = False) -> int:
    """Re-extract Write/Edit artifacts from raw_json of all messages in the DB."""
    cur = conn.cursor()
    cur.execute("DELETE FROM artifacts")
    rows = cur.execute(
        "SELECT message_id, session_id, timestamp, raw_json FROM messages "
        "WHERE tool_names LIKE '%Write%' OR tool_names LIKE '%Edit%'"
    ).fetchall()
    artifacts = []
    for mid, sid, ts, raw in rows:
        if not raw:
            continue
        try:
            event = json.loads(raw)
            content = event.get("message", {}).get("content", [])
            for tool_name, file_path, preview in extract_write_edit_artifacts(content):
                artifacts.append({
                    "session_id": sid,
                    "message_id": mid,
                    "timestamp": ts,
                    "tool_name": tool_name,
                    "file_path": file_path,
                    "content_preview": preview,
                })
        except Exception:
            pass
    if artifacts:
        cur.executemany(
            """INSERT INTO artifacts
               (session_id, message_id, timestamp, tool_name, file_path, content_preview)
               VALUES (:session_id, :message_id, :timestamp, :tool_name, :file_path, :content_preview)""",
            artifacts,
        )
    conn.commit()
    if not quiet:
        print(f"Rebuilt artifacts: {len(artifacts)} records from {len(rows)} messages.")
    return len(artifacts)


def find_spawning_message(conn: sqlite3.Connection, parent_session_id: str,
                          tool_use_id: str | None) -> str | None:
    """Find the parent message whose content contains the Agent tool_use call
    identified by tool_use_id. Returns message_id, or None if not found."""
    if not tool_use_id or not parent_session_id:
        return None
    row = conn.execute(
        "SELECT message_id FROM messages WHERE session_id = ? AND raw_json LIKE ?",
        (parent_session_id, f"%{tool_use_id}%"),
    ).fetchone()
    return row[0] if row else None


def discover_subagent_files(source: Path, parent_session_id: str) -> list[Path]:
    """Find subagents/*.jsonl for a given parent session directory, if any."""
    subdir = source / parent_session_id / "subagents"
    if not subdir.is_dir():
        return []
    return sorted(subdir.glob("agent-*.jsonl"))


def ingest_subagents_for_session(parent_path: Path, conn: sqlite3.Connection,
                                 copy_dest: Path | None, files_dest: Path | None,
                                 project_root: Path,
                                 dry_run: bool, force: bool, quiet: bool) -> tuple[int, int]:
    """Ingest all subagent transcripts for one parent session. Returns (files_processed, messages)."""
    parent_session_id = parent_path.stem
    agent_files = discover_subagent_files(parent_path.parent, parent_session_id)
    if not agent_files:
        return 0, 0

    processed = 0
    total_messages = 0
    for agent_path in agent_files:
        session_id = agent_path.stem
        stat = agent_path.stat()
        if not dry_run and not force:
            row = conn.execute(
                "SELECT source_mtime, source_size FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row and row[0] is not None and row[0] == stat.st_mtime_ns and row[1] == stat.st_size:
                continue

        meta_path = agent_path.with_suffix("").with_suffix(".meta.json")
        agent_type = None
        tool_use_id = None
        spawn_depth = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                agent_type = meta.get("agentType")
                tool_use_id = meta.get("toolUseId")
                spawn_depth = meta.get("spawnDepth")
            except Exception:
                pass

        spawned_from = None if dry_run else find_spawning_message(conn, parent_session_id, tool_use_id)
        stats = ingest_file(
            agent_path, conn if not dry_run else None,
            (copy_dest / "subagents") if copy_dest is not None else None,
            files_dest=files_dest, project_root=project_root, dry_run=dry_run,
            source="subagent", source_mtime=stat.st_mtime_ns, source_size=stat.st_size,
            is_subagent=True, parent_session_id=parent_session_id,
            agent_type=agent_type, spawn_depth=spawn_depth,
            spawned_from_message_id=spawned_from,
            copy_filename=f"{short_uuid(parent_session_id)}-{agent_path.stem}.jsonl",
        )
        processed += 1
        total_messages += stats["messages"]
        if not quiet:
            label = f"{agent_type or '?'}"
            print(f"    subagents/{agent_path.name[:38]:38}  msgs={stats['messages']:5}  "
                  f"type={label}")
    return processed, total_messages


# --- Main ingestion --------------------------------------------------------

def ingest_file(jsonl_path: Path, conn: sqlite3.Connection, copy_dest: Path | None,
                files_dest: Path | None = None, project_root: Path | None = None,
                dry_run: bool = False,
                source: str = "claude", source_mtime: int = 0, source_size: int = 0,
                is_subagent: bool = False, parent_session_id: str | None = None,
                agent_type: str | None = None, spawn_depth: int | None = None,
                spawned_from_message_id: str | None = None,
                copy_filename: str | None = None) -> dict:
    """Ingest one JSONL file. Returns stats dict.

    Subagent transcripts carry the *parent's* sessionId on every event (Claude Code
    quirk) — pass is_subagent=True to force message/attachment session_id to this
    file's own session_id (jsonl_path.stem) instead of trusting the embedded value.
    """
    session_id = jsonl_path.stem
    messages: list[dict] = []
    attachments: list[dict] = []
    title = None
    cwd = None
    entrypoint = None
    version = None
    git_branch = None
    started = None
    ended = None
    skipped = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] {jsonl_path.name}:{lineno} JSON decode error: {e}", file=sys.stderr)
                skipped += 1
                continue
            etype = event.get("type")
            if etype in SKIP_TYPES:
                continue
            cwd = cwd or event.get("cwd")
            entrypoint = entrypoint or event.get("entrypoint")
            version = version or event.get("version")
            git_branch = git_branch or event.get("gitBranch")
            ts = event.get("timestamp")
            if ts:
                if not started or ts < started:
                    started = ts
                if not ended or ts > ended:
                    ended = ts

            if etype in {"user", "assistant"}:
                msg = event.get("message") or {}
                content = msg.get("content")
                msg_uuid = event.get("uuid") or ""
                sid = event.get("sessionId") or session_id
                attachments.extend(
                    collect_attachments_from_content(content, msg_uuid, sid, ts or "")
                )
            path_att = collect_path_attachment_from_event(event)
            if path_att is not None:
                attachments.append(path_att)

            row = parse_message_event(event)
            if row is None:
                continue
            if row["role"] == "ai-title":
                title = row["text_content"] or title
                continue
            row["raw_json"] = line
            messages.append(row)

    if is_subagent:
        for row in messages:
            row["session_id"] = session_id
        for att in attachments:
            att["session_id"] = session_id

    # Collect Write/Edit artifacts from assistant messages
    artifact_rows: list[dict] = []
    for msg in messages:
        if msg["role"] != "assistant" or not msg.get("tool_names"):
            continue
        tnames = msg.get("tool_names") or ""
        if "Write" not in tnames and "Edit" not in tnames:
            continue
        try:
            event = json.loads(msg["raw_json"])
            content = event.get("message", {}).get("content", [])
            for tool_name, file_path, preview in extract_write_edit_artifacts(content):
                artifact_rows.append({
                    "session_id": msg["session_id"],
                    "message_id": msg["message_id"],
                    "timestamp": msg["timestamp"],
                    "tool_name": tool_name,
                    "file_path": file_path,
                    "content_preview": preview,
                })
        except Exception:
            pass

    if dry_run:
        return {
            "session_id": session_id,
            "messages": len(messages),
            "attachments": len(attachments),
            "skipped": skipped,
            "title": title,
            "started": started,
            "ended": ended,
        }

    cur = conn.cursor()
    cur.execute("DELETE FROM artifacts   WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM attachments WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM messages    WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM sessions    WHERE session_id = ?", (session_id,))

    cur.execute(
        """INSERT INTO sessions
           (session_id, started_at, ended_at, title, cwd, entrypoint, version, git_branch,
            message_count, source, source_mtime, source_size,
            parent_session_id, agent_type, spawn_depth, spawned_from_message_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, started, ended, title, cwd, entrypoint, version, git_branch,
         len(messages), source, source_mtime, source_size,
         parent_session_id, agent_type, spawn_depth, spawned_from_message_id)
    )

    cur.executemany(
        """INSERT INTO messages
           (message_id, session_id, parent_id, timestamp, role, type,
            text_content, has_tool_use, tool_names, raw_json)
           VALUES (:message_id, :session_id, :parent_id, :timestamp, :role, :type,
                   :text_content, :has_tool_use, :tool_names, :raw_json)""",
        messages
    )

    if files_dest is not None:
        for rec in attachments:
            try:
                write_attachment_file(rec, files_dest, project_root)
            except Exception as e:
                rec["error"] = f"unexpected: {e}"
        for rec in attachments:
            rec.pop("_data", None)
            rec.pop("_url", None)
            rec.pop("_timestamp", None)
        cur.executemany(
            """INSERT INTO attachments
               (message_id, session_id, position, kind, media_type, source_type,
                original_path, archive_path, sha256, size_bytes, extracted, error)
               VALUES (:message_id, :session_id, :position, :kind, :media_type, :source_type,
                       :original_path, :archive_path, :sha256, :size_bytes, :extracted, :error)""",
            attachments
        )

    if artifact_rows:
        cur.executemany(
            """INSERT INTO artifacts
               (session_id, message_id, timestamp, tool_name, file_path, content_preview)
               VALUES (:session_id, :message_id, :timestamp, :tool_name, :file_path, :content_preview)""",
            artifact_rows,
        )
    conn.commit()

    if copy_dest is not None:
        if copy_filename:
            new_name = copy_filename
        else:
            date = date_from_iso(started or "")
            new_name = f"{date}-{short_uuid(session_id)}.jsonl" if date else f"{session_id}.jsonl"
        dest = copy_dest / new_name
        copy_dest.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != jsonl_path.stat().st_size:
            shutil.copy2(jsonl_path, dest)
        if is_subagent:
            meta_src = jsonl_path.with_suffix("").with_suffix(".meta.json")
            if meta_src.exists():
                meta_dest = dest.with_suffix("").with_suffix(".meta.json")
                if not meta_dest.exists() or meta_dest.stat().st_size != meta_src.stat().st_size:
                    shutil.copy2(meta_src, meta_dest)

    return {
        "session_id": session_id,
        "messages": len(messages),
        "attachments": len(attachments),
        "artifacts": len(artifact_rows),
        "skipped": skipped,
        "title": title,
        "started": started,
        "ended": ended,
    }


def main(config: ProjectConfig, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None,
                        help=f"Source dir with JSONL files (default: {config.claude_source_dir}"
                             + (f" + {len(config.extra_claude_source_dirs)} extra dir(s)"
                                if config.extra_claude_source_dirs else "") + ")")
    parser.add_argument("--db", type=Path, default=config.db_path,
                        help=f"Target SQLite DB (default: {config.db_path})")
    parser.add_argument("--transcripts-dir", type=Path, default=config.transcripts_dir,
                        help=f"Where to copy JSONL (default: {config.transcripts_dir})")
    parser.add_argument("--files-dir", type=Path, default=config.files_dir,
                        help=f"Where to extract attached files (default: {config.files_dir})")
    parser.add_argument("--no-copy", action="store_true", help="Skip copying JSONL to archive/transcripts/")
    parser.add_argument("--no-files", action="store_true", help="Skip extracting attached images/documents")
    parser.add_argument("--dry-run", action="store_true", help="Parse but don't write to DB or copy")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N JSONL files (0 = all)")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest all files, ignoring cached mtime/size")
    parser.add_argument("--rebuild-artifacts", action="store_true",
                        help="Re-scan all raw_json to rebuild artifacts table (run once after upgrade)")
    parser.add_argument("--include-subagents", action="store_true",
                        help="Also ingest <session>/subagents/agent-*.jsonl transcripts "
                             "(source='subagent'; excluded from search by default — see query.py --include-subagents)")
    args = parser.parse_args(argv)

    if args.rebuild_artifacts:
        conn = open_db(args.db)
        n = rebuild_artifacts(conn, quiet=args.quiet)
        conn.close()
        if not args.quiet:
            print(f"Done. {n} artifact records written.")
        return 0

    sources: list[Path] = (
        [args.source] if args.source is not None
        else [config.claude_source_dir] + list(config.extra_claude_source_dirs)
    )
    existing_sources = [s for s in sources if s.is_dir()]
    for s in sources:
        if s not in existing_sources and not args.quiet:
            print(f"[skip] source not found: {s}", file=sys.stderr)
    if not existing_sources:
        print(f"No source dirs found among: {sources}", file=sys.stderr)
        return 1

    files = sorted(f for s in existing_sources for f in s.glob("*.jsonl"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("No JSONL files found.")
        return 0

    if args.dry_run:
        conn = None
        copy_dest = None
        files_dest = None
        if not args.quiet:
            print(f"DRY RUN: would process {len(files)} files")
    else:
        conn = open_db(args.db)
        copy_dest = None if args.no_copy else args.transcripts_dir
        files_dest = None if args.no_files else args.files_dir

    total_messages = 0
    total_attachments = 0
    skipped = 0
    total_agent_files = 0
    total_agent_messages = 0
    for path in files:
        stat = path.stat()
        parent_unchanged = False
        if not args.dry_run and not args.force:
            row = conn.execute(
                "SELECT source_mtime, source_size FROM sessions WHERE session_id = ?",
                (path.stem,)
            ).fetchone()
            if row and row[0] is not None and row[0] == stat.st_mtime_ns and row[1] == stat.st_size:
                skipped += 1
                parent_unchanged = True
                if not args.quiet:
                    print(f"  {path.name[:42]:42}  (unchanged — skipped)")

        if not parent_unchanged:
            if args.dry_run:
                stats = ingest_file(path, None, None, files_dest=None, dry_run=True)
            else:
                stats = ingest_file(path, conn, copy_dest, files_dest=files_dest,
                                    project_root=config.project_root, dry_run=False,
                                    source_mtime=stat.st_mtime_ns, source_size=stat.st_size)
            total_messages += stats["messages"]
            total_attachments += stats.get("attachments", 0)
            if not args.quiet:
                title = (stats.get("title") or "")[:50]
                att = stats.get("attachments", 0)
                att_str = f"  att={att:3}" if att else ""
                print(f"  {path.name[:42]:42}  msgs={stats['messages']:5}{att_str}  "
                      f"started={(stats.get('started') or '')[:19]}  {title}")

        if args.include_subagents:
            n_files, n_msgs = ingest_subagents_for_session(
                path, conn, copy_dest, files_dest, config.project_root,
                args.dry_run, args.force, args.quiet
            )
            total_agent_files += n_files
            total_agent_messages += n_msgs

    if not args.quiet:
        processed = len(files) - skipped
        skip_str = f", {skipped} skipped (unchanged)" if skipped else ""
        att_total = f", {total_attachments} attachments" if total_attachments else ""
        print(f"\nTotal: {len(files)} files ({processed} processed{skip_str}), {total_messages} messages{att_total}")
        if args.include_subagents:
            print(f"Subagents: {total_agent_files} transcripts, {total_agent_messages} messages")
    if conn is not None:
        conn.close()
        last_indexed_file = args.db.parent / ".last_indexed"
        last_indexed_file.write_text(str(time.time()))
    return 0
