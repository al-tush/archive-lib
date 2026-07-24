#!/usr/bin/env python3
"""
Восстановить целиком диалог сессии Claude Code из JSONL-файла в человекочитаемом markdown.

Поиск JSONL ведётся по двум источникам (в порядке):
  1. <config.transcripts_dir>/*.jsonl  — копии в репозитории (имена YYYY-MM-DD-<uuid>.jsonl)
  2. <config.claude_source_dir>/*.jsonl — источник истории

Аргумент UUID — это session_id или его префикс (минимум 4 символа).

Stdlib-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .config import ProjectConfig
from .timeutil import LOCAL_TZ, LOCAL_TZ_LABEL

SKIP_TYPES = {"queue-operation", "last-prompt", "file-history-snapshot", "create", "update"}


def find_jsonl(uuid_prefix: str, dirs: list[Path]) -> Path | None:
    """Find a JSONL file matching uuid_prefix in any of the dirs (first match wins).

    Also checks one level of nesting under each dir:
    - archive copies: <dir>/subagents/*.jsonl, <dir>/copilot/*.jsonl (reconstructed —
      see archive_lib.ingest_copilot.write_session_jsonl)
    - live source:    <dir>/<parent-session>/subagents/*.jsonl
    """
    if len(uuid_prefix) < 4:
        return None
    for d in dirs:
        if not d.is_dir():
            continue
        direct = d / f"{uuid_prefix}.jsonl"
        if direct.exists():
            return direct
        for path in d.glob("*.jsonl"):
            if uuid_prefix in path.stem:
                return path
        for path in d.glob("subagents/*.jsonl"):
            if uuid_prefix in path.stem:
                return path
        for path in d.glob("*/subagents/*.jsonl"):
            if uuid_prefix in path.stem:
                return path
        for path in d.glob("copilot/*.jsonl"):
            if uuid_prefix in path.stem:
                return path
    return None


def fmt_ts(ts: str) -> str:
    if not ts:
        return "?"
    try:
        from datetime import timezone
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S") + f" {LOCAL_TZ_LABEL}"
    except ValueError:
        return ts[:19]


def parse_ts(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def fmt_delta(seconds: float) -> str:
    if seconds < 60:
        return f"+{int(seconds)}s"
    if seconds < 3600:
        return f"+{int(seconds // 60)}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"+{h}h{m}m" if m else f"+{h}h"


PAUSE_THRESHOLD_S = 30 * 60


def _gap_prefix(delta_s: float | None, show_gaps: bool) -> tuple[str, str]:
    """Return (gap_marker, delta_label) to prepend before a rendered event header."""
    if not show_gaps or delta_s is None or delta_s < 1:
        return "", ""
    delta_label = f" {fmt_delta(delta_s)}"
    if delta_s >= PAUSE_THRESHOLD_S:
        h = int(delta_s // 3600)
        m = int((delta_s % 3600) // 60)
        if h and m:
            label = f"{h}h {m}m"
        elif h:
            label = f"{h}h"
        else:
            label = f"{m}m"
        gap_marker = f"\n> ⏸ пауза {label}\n\n"
    else:
        gap_marker = ""
    return gap_marker, delta_label


def format_tool_input(name: str, inp: dict) -> str:
    """Format a tool_use input as a code block."""
    if not isinstance(inp, dict):
        return f"```\n{inp}\n```"
    if name == "Bash" and "command" in inp:
        cmd = inp.get("command", "")
        desc = inp.get("description", "")
        out = f"```bash\n{cmd}\n```"
        if desc:
            out = f"_{desc}_\n\n{out}"
        return out
    return "```json\n" + json.dumps(inp, ensure_ascii=False, indent=2) + "\n```"


def render_content_blocks(blocks, include_thinking: bool) -> list[str]:
    """Render a list of content blocks (text/thinking/tool_use/tool_result) as markdown lines."""
    lines: list[str] = []
    if isinstance(blocks, str):
        lines.append(blocks)
        return lines
    if not isinstance(blocks, list):
        return lines
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            t = block.get("text", "").strip()
            if t:
                lines.append(t)
        elif btype == "thinking":
            if include_thinking:
                t = block.get("thinking", "").strip()
                if t:
                    lines.append("> **[thinking]**")
                    for line in t.splitlines():
                        lines.append(f"> {line}")
        elif btype == "tool_use":
            name = block.get("name", "?")
            tid = (block.get("id") or "")[:12]
            inp = block.get("input") or {}
            lines.append(f"### Tool call: `{name}`  _(id: {tid})_")
            lines.append(format_tool_input(name, inp))
        elif btype == "tool_result":
            tid = (block.get("tool_use_id") or "")[:12]
            content = block.get("content")
            is_error = block.get("is_error")
            header = f"### Tool result  _(id: {tid}{', ERROR' if is_error else ''})_"
            lines.append(header)
            if isinstance(content, str):
                lines.append("```\n" + content + "\n```")
            elif isinstance(content, list):
                for sub in content:
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        st = sub.get("text", "")
                        lines.append("```\n" + st + "\n```")
        elif btype in {"image", "document"}:
            source = block.get("source") or {}
            stype = source.get("type", "?")
            media_type = source.get("media_type") or "?"
            if stype == "base64":
                data = source.get("data") or ""
                size_kb = round(len(data) * 3 / 4 / 1024)
                lines.append(f"_[{btype}: {media_type}, base64 ~{size_kb} KB]_")
            elif stype == "url":
                url = source.get("url") or "?"
                lines.append(f"_[{btype}: {media_type}, url: {url}]_")
            elif stype == "file":
                path = source.get("path") or source.get("file_id") or "?"
                lines.append(f"_[{btype}: {media_type}, file: {path}]_")
            else:
                lines.append(f"_[{btype}: {media_type}, source_type={stype}]_")
    return lines


def classify_user_content(content) -> str:
    """Return 'text', 'tool_result', or 'mixed' depending on what blocks the user message contains.

    In the Anthropic API, role='user' carries both real human input (text blocks) and
    automated tool responses (tool_result blocks). Distinguishing them in the rendered output
    avoids confusion where 'user' looks like 'human said this'.
    """
    if isinstance(content, str):
        return "text"
    if not isinstance(content, list):
        return "text"
    has_text = False
    has_tool_result = False
    for block in content:
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        if bt == "text":
            has_text = True
        elif bt == "tool_result":
            has_tool_result = True
    if has_tool_result and not has_text:
        return "tool_result"
    if has_text and has_tool_result:
        return "mixed"
    return "text"


def render_event(event: dict, include_thinking: bool, include_attachments: bool,
                 delta_s: float | None = None, show_gaps: bool = True) -> str | None:
    etype = event.get("type")
    if etype in SKIP_TYPES:
        return None

    ts = fmt_ts(event.get("timestamp", ""))
    gap_marker, delta_label = _gap_prefix(delta_s, show_gaps)

    if etype == "user":
        msg = event.get("message") or {}
        content = msg.get("content")
        body = render_content_blocks(content, include_thinking)
        if not body:
            return None
        kind = classify_user_content(content)
        if kind == "tool_result":
            label = "tool_result _(automated reply to assistant's tool call)_"
        elif kind == "mixed":
            label = "user + tool_result _(human input + tool response)_"
        else:
            label = "user"
        return f"{gap_marker}## [{ts}{delta_label}] {label}\n\n" + "\n\n".join(body) + "\n"

    if etype == "assistant":
        msg = event.get("message") or {}
        content = msg.get("content")
        model = msg.get("model", "")
        body = render_content_blocks(content, include_thinking)
        if not body:
            return None
        header = f"{gap_marker}## [{ts}{delta_label}] assistant"
        if model:
            header += f"  _({model})_"
        return header + "\n\n" + "\n\n".join(body) + "\n"

    if etype == "tool_result":
        msg = event.get("message") or {}
        body = render_content_blocks(msg.get("content"), include_thinking)
        if not body:
            r = event.get("toolUseResult")
            if isinstance(r, str) and r.strip():
                body = ["```\n" + r + "\n```"]
            elif isinstance(r, dict):
                t = r.get("content") or r.get("text") or r.get("stdout") or ""
                if isinstance(t, str) and t.strip():
                    body = ["```\n" + t + "\n```"]
        if not body:
            return None
        return f"{gap_marker}## [{ts}{delta_label}] tool_result\n\n" + "\n\n".join(body) + "\n"

    if etype == "attachment":
        if not include_attachments:
            return None
        att = event.get("attachment") or {}
        name = att.get("hookName") or att.get("type") or "attachment"
        stdout = att.get("stdout") or ""
        content = att.get("content") or ""
        body = ""
        if isinstance(stdout, str) and stdout.strip():
            body = stdout
        elif isinstance(content, str) and content.strip():
            body = content
        if not body:
            return None
        return f"{gap_marker}## [{ts}{delta_label}] attachment _({name})_\n\n```\n{body}\n```\n"

    if etype == "system":
        msg = event.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            body = [content]
        else:
            body = render_content_blocks(content, include_thinking)
        if not body:
            return None
        return f"{gap_marker}## [{ts}{delta_label}] system\n\n" + "\n\n".join(body) + "\n"

    if etype == "ai-title":
        title = event.get("title", "")
        if title:
            return f"_(ai-title: {title})_\n"
        return None

    return f"{gap_marker}## [{ts}{delta_label}] {etype} _(unhandled type)_\n\n```json\n{json.dumps(event, ensure_ascii=False, indent=2)[:2000]}\n```\n"


def render_session(jsonl_path: Path, include_thinking: bool, include_attachments: bool,
                   show_gaps: bool = True) -> str:
    events = []
    started = None
    ended = None
    cwd = None
    entrypoint = None
    version = None
    git_branch = None
    title = None
    session_id = jsonl_path.stem
    if "-" in session_id and len(session_id) > 36:
        session_id = session_id[-36:]
    agent_id = None
    attribution_agent = None

    with jsonl_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as e:
                events.append(f"_(line {lineno}: JSON decode error: {e})_\n")
                continue
            ts = event.get("timestamp")
            if ts:
                if not started or ts < started:
                    started = ts
                if not ended or ts > ended:
                    ended = ts
            cwd = cwd or event.get("cwd")
            entrypoint = entrypoint or event.get("entrypoint")
            version = version or event.get("version")
            git_branch = git_branch or event.get("gitBranch")
            if event.get("isSidechain"):
                agent_id = agent_id or event.get("agentId")
                attribution_agent = attribution_agent or event.get("attributionAgent")
            sid = event.get("sessionId")
            if sid and len(sid) == 36:
                session_id = sid
            if event.get("type") == "ai-title":
                title = event.get("title") or title
            events.append(event)

    out: list[str] = []
    if agent_id:
        out.append(f"# Subagent `{agent_id}`\n")
    else:
        out.append(f"# Session `{session_id}`\n")
    if title:
        out.append(f"**{title}**\n")
    meta = []
    if agent_id:
        meta.append(f"- Parent session: `{session_id}`")
        if attribution_agent:
            meta.append(f"- Agent type: `{attribution_agent}`")
        meta_path = jsonl_path.with_suffix("").with_suffix(".meta.json")
        if meta_path.exists():
            try:
                sidecar = json.loads(meta_path.read_text(encoding="utf-8"))
                if sidecar.get("description"):
                    meta.append(f"- Task: {sidecar['description']}")
                if sidecar.get("toolUseId"):
                    meta.append(f"- Spawned by tool_use: `{sidecar['toolUseId']}`"
                                f" — `python restore.py {session_id[:8]}` for parent context")
            except Exception:
                pass
    if started:
        meta.append(f"- Started: `{fmt_ts(started)}`")
    if ended:
        meta.append(f"- Ended: `{fmt_ts(ended)}`")
    if cwd:
        meta.append(f"- Cwd: `{cwd}`")
    if entrypoint:
        meta.append(f"- Entrypoint: `{entrypoint}`")
    if version:
        meta.append(f"- Version: `{version}`")
    if git_branch:
        meta.append(f"- Branch: `{git_branch}`")
    meta.append(f"- Source: `{jsonl_path}`")
    out.append("\n".join(meta))
    out.append("\n---\n")

    prev_dt: datetime | None = None
    for event in events:
        if isinstance(event, str):
            out.append(event)
            continue
        curr_dt = parse_ts(event.get("timestamp", ""))
        delta_s: float | None = None
        if curr_dt is not None and prev_dt is not None:
            delta_s = max(0.0, (curr_dt - prev_dt).total_seconds())
        rendered = render_event(event, include_thinking, include_attachments,
                                delta_s=delta_s, show_gaps=show_gaps)
        if rendered:
            out.append(rendered)
            out.append("---\n")
            if curr_dt is not None:
                prev_dt = curr_dt

    return "\n".join(out)


def main(config: ProjectConfig, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("uuid", help="Session UUID или префикс (>=4 символа)")
    parser.add_argument("-o", "--output", type=Path, help="Сохранить в файл (по умолчанию stdout)")
    parser.add_argument("--thinking", action="store_true", help="Включить thinking-блоки")
    parser.add_argument("--no-attachments", action="store_true",
                        help="Пропустить attachment-события (hook outputs)")
    parser.add_argument("--no-gaps", action="store_true",
                        help="Не показывать дельты времени и маркеры пауз")
    parser.add_argument("--raw", action="store_true", help="Просто вывести содержимое JSONL без рендера")
    parser.add_argument("--archive-dir", type=Path, default=config.transcripts_dir)
    parser.add_argument("--source-dir", type=Path, default=config.claude_source_dir)
    args = parser.parse_args(argv)

    search_dirs = [args.archive_dir, args.source_dir] + list(config.extra_claude_source_dirs)
    jsonl_path = find_jsonl(args.uuid, search_dirs)
    if not jsonl_path:
        listed = "\n  ".join(str(d) for d in search_dirs)
        print(f"JSONL не найден для префикса '{args.uuid}' в:\n  {listed}", file=sys.stderr)
        return 1

    if args.raw:
        content = jsonl_path.read_text(encoding="utf-8")
    else:
        content = render_session(jsonl_path, include_thinking=args.thinking,
                                 include_attachments=not args.no_attachments,
                                 show_gaps=not args.no_gaps)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Записано в {args.output}  ({len(content):,} символов)", file=sys.stderr)
    else:
        sys.stdout.write(content)
    return 0
