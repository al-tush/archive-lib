#!/usr/bin/env python3
"""
Систематический полный обход (full-scan) кандидатов-сессий по запросу —
альтернатива top-k для вопросов, где релевантность отдельного сниппета и факт
"что происходило системно/по времени" — разные оси, которые score не покрывает
(подсчёт/агрегация по корпусу, хронологический multi-hop синтез). См.
apteki/тз_rag_full_scan_rlm_обход.md и apteki/apteki_rag_full_scan_эксперимент_
2026-10-10.md — там же объяснение, почему top-k категориально не отвечает на
такие вопросы, не просто "слабее".

Чистый механический слой — никаких вызовов LM. Решение "позвать ли субагента на
батч" и сам синтез остаются на стороне вызывающей Claude Code сессии (см.
`_run_multi_archive`-style separation уже принятое в query.py: один модуль
отвечает за один DB, цикл по архивам/батчам — на стороне CLI-обёртки).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from .config import ProjectConfig
from . import query as query_mod
from . import restore as restore_mod

# Deliberately not a real cap — the whole point of this module is to NOT
# apply top-k. Bounded in practice by how many rows actually match.
_NO_CAP = 1_000_000


@dataclass
class SessionCandidate:
    session_id: str
    source: str
    first_hit_ts: str
    hit_count: int
    best_score: Optional[float]


# Фразы-признаки класса B/C (подсчёт/агрегация, хронологический синтез,
# исчерпывающее перечисление) — см. apteki/archive/rag_reference.md, раздел
# "Full-scan vs top-k: когда переключаться". Единственное место хранения этого
# списка — query.py's _full_scan_hint() импортирует его отсюда, документация
# ссылается на этот файл, а не повторяет список.
# Это НЕ автоклассификатор, принимающий решение (см. тз_rag_full_scan_rlm_обход.md
# п.3) — только сигнал-подсказка в выводе query.py; переключение на full_scan.py
# остаётся за вызывающей стороной.
EXHAUSTIVE_QUERY_PATTERNS = [
    r"\bвсе\b", r"\bвсех\b", r"\bвсём\b", r"\bвсему\b",
    r"\bкажд\w*\b",
    r"сколько раз", r"как часто",
    r"как менял\w*", r"эволюци\w*",
    r"хронолог\w*",
    r"за период", r"за последн\w+ (?:недел|месяц|год|день|дн)",
]
_EXHAUSTIVE_RE = re.compile("|".join(EXHAUSTIVE_QUERY_PATTERNS), re.IGNORECASE)


def looks_exhaustive(query_text: str) -> bool:
    """Дешёвая эвристика по тексту запроса (не NLP, не LM) — похож ли вопрос на
    класс B/C по формулировке. Ложноположительные дешёвы (лишняя строка hint),
    ложноотрицательные — тот же риск, что и у чисто документационного правила,
    просто теперь видимый в коде и легко расширяемый одним списком."""
    return bool(_EXHAUSTIVE_RE.search(query_text))


def _fake_args(fts_query: str, *, semantic: bool, hybrid: bool,
                roles: Optional[str], from_date: Optional[str],
                to_date: Optional[str]) -> SimpleNamespace:
    return SimpleNamespace(
        query=fts_query,
        session=None,
        all_roles=False,
        role=None,
        roles=roles,
        from_date=from_date,
        to_date=to_date,
        include_excluded=False,
        include_code=False,
        include_rag_echo=False,
        has_tool_use=False,
        source=None,
        include_subagents=False,
        limit=_NO_CAP,
        semantic=semantic,
        hybrid=hybrid,
        corpus="transcripts",  # full-scan — сессии-транскрипты, не файлы проекта
    )


def enumerate_candidates(
    conn: sqlite3.Connection,
    fts_query: str,
    *,
    semantic: bool = False,
    hybrid: bool = False,
    roles: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    source_label: str = "claude",
    config: Optional[ProjectConfig] = None,
) -> list[SessionCandidate]:
    """Перечислить ВСЕ сессии-кандидаты по запросу в этой БД (без top-k cap),
    по одной записи на session_id: first_hit_ts, hit_count, best_score.

    Один DB-коннект на вызов, как и query.search()/search_hybrid()/
    search_semantic() — цикл по нескольким архивам (--archives) не встроен
    сюда намеренно: query.py уже решает эту задачу на уровне CLI
    (`_run_multi_archive`), дублировать её внутри библиотечной функции было бы
    второй реализацией того же цикла. Обёртка CLI этого модуля (см. main())
    делает то же самое для full-scan.

    `fts_query` должен быть уже санитизирован/расширен (как делает query.py's
    main() перед вызовом search()) — эта функция сама этим не занимается,
    чтобы не дублировать ту логику.
    """
    if (semantic or hybrid) and config is None:
        raise ValueError("config обязателен для semantic/hybrid (нужен model_name)")

    args = _fake_args(fts_query, semantic=semantic, hybrid=hybrid,
                       roles=roles, from_date=from_date, to_date=to_date)

    if hybrid:
        results = query_mod.search_hybrid(conn, args, config)
    elif semantic:
        results = query_mod.search_semantic(conn, args, config)
    else:
        results = query_mod.search(conn, args)

    by_session: dict[str, SessionCandidate] = {}
    for r in results:
        sid = r.get("session_id")
        if not sid:
            continue
        ts = r.get("timestamp") or ""
        score = r.get("score")
        if score is None:
            rank = r.get("rank")  # bm25: lower is better
            score = -rank if rank is not None else None
        existing = by_session.get(sid)
        if existing is None:
            by_session[sid] = SessionCandidate(
                session_id=sid, source=source_label,
                first_hit_ts=ts, hit_count=1, best_score=score,
            )
        else:
            existing.hit_count += 1
            if ts and (not existing.first_hit_ts or ts < existing.first_hit_ts):
                existing.first_hit_ts = ts
            if score is not None and (existing.best_score is None or score > existing.best_score):
                existing.best_score = score
    return list(by_session.values())


def chronological_batches(candidates: list[SessionCandidate],
                           batch_size: int = 8) -> list[list[SessionCandidate]]:
    """Сортирует по first_hit_ts, режет на батчи по batch_size — каждый батч
    непрерывный кусок хронологии, не случайная выборка по score."""
    ordered = sorted(candidates, key=lambda c: c.first_hit_ts or "")
    return [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]


def restore_batch(config: ProjectConfig, batch: list[SessionCandidate],
                   include_thinking: bool = False) -> str:
    """Конкатенирует restore.py-эквивалент (полный текст сессии) для сессий
    одного батча, с явными разделителями между сессиями."""
    search_dirs = [config.transcripts_dir, config.claude_source_dir] + list(config.extra_claude_source_dirs)
    parts = []
    for cand in batch:
        # 8-char prefix, not the full UUID: find_jsonl() matches by substring
        # against the filename stem, and an archived/rotated copy is named
        # YYYY-MM-DD-<8char>.jsonl — the full 36-char UUID is never a
        # substring of that, so passing it silently misses every session
        # whose JSONL has already rotated out of the live source dir (found
        # by subagent validation 10.10.2026: batches covering dates older
        # than ~1 month all came back "not found").
        jsonl_path = restore_mod.find_jsonl(cand.session_id[:8], search_dirs)
        if jsonl_path is None:
            parts.append(f"# Session `{cand.session_id}`\n\n_(JSONL не найден — сессия вне текущих источников)_\n")
            continue
        parts.append(restore_mod.render_session(
            jsonl_path, include_thinking=include_thinking, include_attachments=False,
        ))
    return ("\n\n" + "=" * 80 + "\n\n").join(parts)


def main(config: ProjectConfig, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="Запрос (FTS5 синтаксис, как у query.py)")
    parser.add_argument("--db", type=Path, default=config.db_path)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Размер батча сессий на один restore/субагент (по умолчанию: 8)")
    parser.add_argument("--hybrid", action="store_true",
                        help="Семантический ранкинг кандидатов внутри запроса "
                             "(хронология батчей всё равно по first_hit_ts, не по score)")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--roles", metavar="ROLE[,ROLE]")
    parser.add_argument("--from", dest="from_date", metavar="YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", metavar="YYYY-MM-DD")
    parser.add_argument("--no-expand", action="store_true",
                        help="Отключить авторасширение кириллических слов (как у query.py)")
    parser.add_argument("--manifest", action="store_true",
                        help="Печатать JSON-манифест батчей вместо текста, даже если "
                             "кандидатов <= batch-size")
    args = parser.parse_args(argv)

    if args.semantic or args.hybrid:
        from .pyenv import ensure_semantic_capable
        ensure_semantic_capable()

    if not args.db.exists():
        print(f"DB not found: {args.db}\nRun ingest.py first.", file=sys.stderr)
        return 1

    fts_query = args.query
    if not args.semantic and not args.no_expand:
        fts_query = query_mod._sanitize_fts_query(fts_query)
        expanded = query_mod._expand_cyrillic(fts_query)
        if expanded != fts_query:
            print(f"[expand] {expanded}", file=sys.stderr)
            fts_query = expanded

    conn = sqlite3.connect(args.db)
    try:
        candidates = enumerate_candidates(
            conn, fts_query, semantic=args.semantic, hybrid=args.hybrid,
            roles=args.roles, from_date=args.from_date, to_date=args.to_date,
            config=config,
        )
    finally:
        conn.close()

    if not candidates:
        print("(нет кандидатов)", file=sys.stderr)
        return 0

    batches = chronological_batches(candidates, batch_size=args.batch_size)
    total_hits = sum(c.hit_count for c in candidates)
    print(
        f"[full-scan] {len(candidates)} сессий-кандидатов, {total_hits} хитов, "
        f"{len(batches)} батч(ей) по {args.batch_size}",
        file=sys.stderr,
    )

    if args.manifest or len(batches) > 1:
        manifest = {
            "query": args.query,
            "batch_size": args.batch_size,
            "candidate_count": len(candidates),
            "batches": [
                {
                    "batch_index": i,
                    "session_ids": [c.session_id for c in batch],
                    "date_range": [batch[0].first_hit_ts, batch[-1].first_hit_ts],
                }
                for i, batch in enumerate(batches)
            ],
            "next_step": (
                "На каждый батч — Agent tool с задачей и списком session_ids; "
                "на каждую сессию в батче — restore.py <session_id> целиком; "
                "итог слить синтезом (хронологический multi-hop), не механически."
            ),
        }
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(restore_batch(config, candidates))
    return 0
