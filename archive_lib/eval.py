#!/usr/bin/env python3
"""
Golden-set evaluation for semantic RAG.

Runs each query in config.golden_set, checks if expected content appears in
top-K results, reports Recall@K and MRR (Mean Reciprocal Rank).

Calls query.search_semantic() directly (rather than reimplementing pooling/
scoring here) so eval always measures the actual production search path.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from .config import ProjectConfig
from . import query as query_mod
from . import full_scan as full_scan_mod


class _FakeArgs:
    query = ""
    session = None
    all_roles = False
    role = None
    roles = None
    from_date = None
    to_date = None
    include_excluded = False
    include_code = False
    include_rag_echo = False
    has_tool_use = False
    include_subagents = False
    source = None
    limit = 5
    corpus = "all"


def run_eval(conn: sqlite3.Connection, config: ProjectConfig, k: int, verbose: bool) -> int:
    golden_set = config.golden_set
    if not golden_set:
        print(f"(no golden set configured for project '{config.name}')")
        return 0

    hits = 0
    reciprocal_ranks: list[float] = []
    args = _FakeArgs()
    args.limit = k

    for query, expected, desc in golden_set:
        args.query = query
        try:
            results = query_mod.search_semantic(conn, args, config)
        except Exception as e:
            print(f"  [ERROR] {desc}: {e}")
            reciprocal_ranks.append(0.0)
            continue

        found_at = None
        for rank, r in enumerate(results, 1):
            text = (r.get("text_content") or "").lower()
            path = (r.get("path") or "").lower()
            if expected.lower() in text or expected.lower() in path:
                found_at = rank
                break

        if found_at:
            hits += 1
            reciprocal_ranks.append(1.0 / found_at)
        else:
            reciprocal_ranks.append(0.0)

        status = f"HIT@{found_at}" if found_at else "MISS"
        print(f"  {status:8}  {desc}")

        if verbose:
            for i, r in enumerate(results[:3], 1):
                label = r.get("path") or (r.get("session_id") or "")[:8]
                snippet = (r.get("text_content") or "")[:120].replace("\n", " ")
                print(f"           {i}. [{label}] score={r['score']:.3f}  {snippet}")
            print()

    n = len(golden_set)
    recall = hits / n
    mrr = sum(reciprocal_ranks) / n

    print(f"\n{'─'*50}")
    print(f"Recall@{k}: {recall:.2%}  ({hits}/{n} queries)")
    print(f"MRR@{k}:    {mrr:.3f}")
    return 0


def run_full_scan_eval(conn: sqlite3.Connection, config: ProjectConfig, verbose: bool) -> int:
    """Compare baseline (hybrid top-10) vs full-scan candidate coverage per
    config.full_scan_golden_set entry.

    Deliberately NOT a Recall@K/MRR number: for full-scan's target questions
    (counting/chronological synthesis) there is no single right chunk to land
    in top-K, so "hit" isn't well-defined the way run_eval() defines it. What
    this prints instead — baseline vs full-scan candidate counts, and each
    entry's expected_facts — is raw material for an LM-judge pass (whether
    each expected fact actually shows up in a synthesis built from baseline
    vs from the full-scan candidate set), which stays outside this function:
    eval.py is plain Python without an API key or agentic loop, same
    constraint full_scan.py itself is built around (see
    apteki/тз_rag_full_scan_rlm_обход.md п.3/п.10) — the judging call belongs
    to whoever runs this (a human, or a Claude Code session via Agent tool).
    """
    golden_set = config.full_scan_golden_set
    if not golden_set:
        print(f"(no full-scan golden set configured for project '{config.name}')")
        return 0

    for entry in golden_set:
        query_str = entry["query"]
        cls = entry.get("class", "?")
        expected_facts = entry.get("expected_facts") or []

        print(f"\n{'─'*60}")
        print(f"[{cls}] {query_str}")

        fts_query = query_mod._expand_cyrillic(query_mod._sanitize_fts_query(query_str))

        baseline_args = _FakeArgs()
        baseline_args.query = fts_query
        baseline_args.limit = 10
        try:
            baseline_results = query_mod.search_hybrid(conn, baseline_args, config)
        except Exception as e:
            print(f"  baseline [ERROR] {e}")
            baseline_results = []
        baseline_sessions = {r["session_id"] for r in baseline_results if r.get("session_id")}

        candidates = full_scan_mod.enumerate_candidates(conn, fts_query)
        batches = full_scan_mod.chronological_batches(candidates, batch_size=8)

        print(f"  baseline (hybrid top-10): {len(baseline_sessions)} сессий")
        print(f"  full-scan:                {len(candidates)} кандидатов, {len(batches)} батч(ей)")

        if expected_facts:
            print("  expected_facts (для LM-judge — не проверяется механически здесь):")
            for fact in expected_facts:
                print(f"    - {fact}")
        else:
            print("  (нет expected_facts — вопрос сформулирован, но не валидирован полным "
                  "full-scan-прогоном; см. apteki_rag_full_scan_эксперимент_2026-10-10.md)")

        if verbose:
            for i, c in enumerate(candidates[:5], 1):
                print(f"    {i}. {c.session_id[:8]}  {c.first_hit_ts}")

    print(f"\n{'─'*60}")
    print("Не Recall@K/MRR — чтобы сравнить, сколько из expected_facts присутствует в "
          "синтезе по full-scan vs по baseline, нужен отдельный LM-judge (Agent tool); "
          "eval.py сам LM не вызывает.")
    return 0


def main(config: ProjectConfig, argv: list[str]) -> int:
    default_k = sum(config.pool_guarantees.values())
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["semantic", "full-scan"], default="semantic",
                        help="semantic: Recall@K/MRR on config.golden_set (default). "
                             "full-scan: baseline-vs-full-scan coverage on "
                             "config.full_scan_golden_set (see run_full_scan_eval docstring)")
    parser.add_argument("--k", type=int, default=default_k,
                        help=f"top-K to check (default: {default_k}, the sum of this "
                             "project's pool_guarantees)")
    parser.add_argument("--verbose", action="store_true", help="show top-3 snippets per query")
    parser.add_argument("--db", type=Path, default=config.db_path)
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"DB not found: {args.db}\nRun !update.py first.", file=sys.stderr)
        return 1

    if args.mode == "full-scan":
        conn = sqlite3.connect(args.db)
        try:
            return run_full_scan_eval(conn, config, args.verbose)
        finally:
            conn.close()

    print("loading model …")
    conn = sqlite3.connect(args.db)
    try:
        return run_eval(conn, config, args.k, args.verbose)
    finally:
        conn.close()
