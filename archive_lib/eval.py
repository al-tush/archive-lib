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


class _FakeArgs:
    query = ""
    all_roles = False
    role = None
    roles = None
    from_date = None
    to_date = None
    include_excluded = False
    include_code = False
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


def main(config: ProjectConfig, argv: list[str]) -> int:
    default_k = sum(config.pool_guarantees.values())
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=default_k,
                        help=f"top-K to check (default: {default_k}, the sum of this "
                             "project's pool_guarantees)")
    parser.add_argument("--verbose", action="store_true", help="show top-3 snippets per query")
    parser.add_argument("--db", type=Path, default=config.db_path)
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"DB not found: {args.db}\nRun !update.py first.", file=sys.stderr)
        return 1

    print("loading model …")
    conn = sqlite3.connect(args.db)
    try:
        return run_eval(conn, config, args.k, args.verbose)
    finally:
        conn.close()
