#!/usr/bin/env python3
"""
Считает время, проведённое с ИИ за каждый день.

Источники данных: transcripts.db каждого проекта (claude + copilot сессии),
из archive_lib.registry.KNOWN_ARCHIVES — то есть по умолчанию по всем
зарегистрированным проектам.

Правила:
- Пауза ≥ 5 минут между сообщениями = перерыв (конец активного окна)
- Перекрывающаяся активность в разных сессиях и проектах в одно время не суммируется
  (все временные метки сливаются в единый таймлайн перед подсчётом)

Использование:
  python3 ai_time.py                         # все проекты, все дни
  python3 ai_time.py --from 2026-06-01       # с указанной даты
  python3 ai_time.py --days 7                # последние N дней
  python3 ai_time.py --project project-a     # только один проект
  python3 ai_time.py --weekly                # добавить итоги по неделям
"""

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .registry import KNOWN_ARCHIVES as DATABASES

IDLE_THRESHOLD = timedelta(minutes=5)


def load_timestamps(db_path: Path) -> tuple[list[datetime], dict[str, int]]:
    """Читает все временные метки из БД. Возвращает (timestamps, counts_by_source)."""
    timestamps = []
    counts: dict[str, int] = defaultdict(int)
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT m.timestamp, s.source FROM messages m "
        "JOIN sessions s USING(session_id) "
        "WHERE m.timestamp != ''"
    ).fetchall()
    con.close()
    for ts_str, source in rows:
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            timestamps.append(dt)
            counts[source] += 1
        except Exception:
            pass
    return timestamps, counts


def compute_active_windows(timestamps: list[datetime]) -> list[tuple[datetime, datetime]]:
    """
    Из отсортированного списка временных меток строит список активных окон.
    Окно заканчивается, когда пауза ≥ IDLE_THRESHOLD.
    """
    if not timestamps:
        return []
    sorted_ts = sorted(timestamps)
    windows = []
    win_start = win_end = sorted_ts[0]
    for ts in sorted_ts[1:]:
        if ts - win_end < IDLE_THRESHOLD:
            win_end = ts
        else:
            windows.append((win_start, win_end))
            win_start = win_end = ts
    windows.append((win_start, win_end))
    return windows


def merge_windows(windows: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """Сливает перекрывающиеся и смежные окна."""
    if not windows:
        return []
    sorted_w = sorted(windows, key=lambda w: w[0])
    merged = [sorted_w[0]]
    for start, end in sorted_w[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + IDLE_THRESHOLD:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def windows_to_minutes_by_day(windows: list[tuple[datetime, datetime]]) -> dict[str, float]:
    """
    Суммирует активные минуты по дням (UTC+3).
    Окно, перекрывающее полночь, делится между днями.
    """
    TZ_OFFSET = timedelta(hours=3)
    day_minutes: dict[str, float] = defaultdict(float)
    for start, end in windows:
        cur = start + TZ_OFFSET
        end_local = end + TZ_OFFSET
        while True:
            day_key = cur.strftime("%Y-%m-%d")
            midnight = cur.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            chunk_end = min(end_local, midnight)
            duration = (chunk_end - cur).total_seconds() / 60
            if duration > 0:
                day_minutes[day_key] += duration
            if chunk_end >= end_local:
                break
            cur = midnight
    return day_minutes


def format_minutes(total_min: float) -> str:
    h, m = int(total_min) // 60, int(total_min) % 60
    return "%dч %02dмин" % (h, m) if h else "%dмин" % m


def iso_week(day_str: str):
    d = datetime.strptime(day_str, "%Y-%m-%d")
    y, w, _ = d.isocalendar()
    return (y, w)


def week_label(day_str: str) -> str:
    d = datetime.strptime(day_str, "%Y-%m-%d")
    mon = d - timedelta(days=d.weekday())
    sun = mon + timedelta(days=6)
    return "%s – %s" % (mon.strftime("%d.%m"), sun.strftime("%d.%m"))


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Время с ИИ по дням")
    parser.add_argument("--from", dest="from_date", metavar="YYYY-MM-DD",
                        help="Показывать дни начиная с этой даты")
    parser.add_argument("--days", type=int, metavar="N",
                        help="Последние N дней")
    parser.add_argument("--project", choices=list(DATABASES.keys()),
                        help="Только указанный проект")
    parser.add_argument("--weekly", action="store_true",
                        help="Добавить итоги по неделям")
    args = parser.parse_args(argv)

    dbs_to_use = (
        {args.project: DATABASES[args.project]}
        if args.project
        else DATABASES
    )

    print("Загружаю данные...", file=sys.stderr)
    all_timestamps: list[datetime] = []
    for proj_name, db_path in dbs_to_use.items():
        if not db_path.exists():
            print("  [warn] не найдена БД: %s" % db_path, file=sys.stderr)
            continue
        ts, counts = load_timestamps(db_path)
        sources = ", ".join("%s=%d" % (k, v) for k, v in sorted(counts.items()))
        print("  %s: %d меток (%s)" % (proj_name, len(ts), sources), file=sys.stderr)
        all_timestamps.extend(ts)

    if not all_timestamps:
        print("Нет данных.")
        return

    windows = compute_active_windows(all_timestamps)
    merged = merge_windows(windows)
    day_minutes = windows_to_minutes_by_day(merged)

    cutoff = None
    if args.from_date:
        cutoff = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    elif args.days:
        cutoff = (datetime.now().date() - timedelta(days=args.days - 1))

    days = sorted(day_minutes.keys())
    if cutoff:
        days = [d for d in days if d >= str(cutoff)]

    print("%-12s  %-10s  бар (1 блок = 5 мин)" % ("Дата", "Время"))
    print("─" * 58)

    total = 0.0
    cur_week = None
    week_total = 0.0
    first_day_of_week = None

    for day in days:
        mins = day_minutes[day]
        wk = iso_week(day)

        if args.weekly:
            if cur_week is None:
                cur_week = wk
                first_day_of_week = day
            if wk != cur_week:
                print("  └─ W%02d %-14s итого %s" % (
                    cur_week[1], week_label(first_day_of_week), format_minutes(week_total)))
                print()
                cur_week = wk
                week_total = 0.0
                first_day_of_week = day

        bar = "█" * min(int(mins / 5), 44)
        print("%-12s  %-10s  %s" % (day, format_minutes(mins), bar))
        week_total += mins
        total += mins

    if args.weekly and cur_week:
        print("  └─ W%02d %-14s итого %s" % (
            cur_week[1], week_label(first_day_of_week), format_minutes(week_total)))

    print()
    print("─" * 58)
    print("Всего:          %s  (%d активных дней)" % (format_minutes(total), len(days)))
    if days:
        print("Среднее/день:   %s" % format_minutes(total / len(days)))
    print("(порог паузы: 5 мин; часовой пояс: UTC+3)")
