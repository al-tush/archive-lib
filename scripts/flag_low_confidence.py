#!/usr/bin/env python3
"""
Разбирает JSON от `whisper-cli -ojf` (--output-json-full) и печатает токены
с вероятностью ниже порога — кандидаты на ручную сверку.

Обоснование и мини-регрессионный тест (3 известных случая — «кастировать»,
«мэров», «вези на» — whisper.cpp) — apteki/распознавание_речи_diarization_
альтернативы.md, раздел «ТЗ: точность распознавания…» (20.08.2026).
Вывод: на воспроизведённых ошибках токен с ошибкой (или граница слова рядом)
стабильно даёт p ниже общего фона транскрипта (0.06–0.25 на ошибке против
типичных 0.85–1.0 на верном тексте) — сигнал реальный, но не идеальный
(единичные корректные слова тоже могут давать локальный провал p ~0.5).
Порог по умолчанию 0.6 — эмпирически разделяет известные ошибки от фона на
малой выборке (n=3), не откалиброван на большом корпусе.

Использование:
    whisper-cli -m model.bin -f audio.wav -ojf -of out -np
    python3 flag_low_confidence.py out.json [--threshold 0.6] [--context 3]
"""
import argparse
import json
import sys


def flag(json_path: str, threshold: float, context: int) -> None:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    tokens = []
    for seg in data["transcription"]:
        tokens.extend(seg["tokens"])

    flagged = [
        i for i, t in enumerate(tokens)
        if not t["text"].startswith("[_") and t.get("p", 1.0) < threshold
    ]

    if not flagged:
        print(f"Низкоуверенных токенов (p < {threshold}) не найдено.")
        return

    print(f"Найдено {len(flagged)} токенов с p < {threshold}:\n")
    for i in flagged:
        lo, hi = max(0, i - context), min(len(tokens), i + context + 1)
        ctx = "".join(
            f"[{t['text']}]" if j == i else t["text"]
            for j, t in enumerate(tokens[lo:hi], start=lo)
        )
        print(f"  p={tokens[i]['p']:.3f}  {ctx.strip()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--context", type=int, default=3, help="токенов контекста с каждой стороны")
    args = ap.parse_args()
    flag(args.json_path, args.threshold, args.context)
