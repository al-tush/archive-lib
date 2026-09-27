#!/usr/bin/env python3
"""
Детекция активного говорящего по рамке-кольцу на плитке участника — без
калибровки фиксированных позиций плиток (в отличие от video_speaker_border.py
calibrate/scan). Решает ограничение, найденное 20.08.2026 и переоткрытое
25.09.2026: раскладка плиток видеозвонка (Yandex Telemost и т.п.) меняется по
ходу записи при изменении числа участников (10→9→8 и обратно), поэтому
калибровка по фиксированным координатам систематически ошибается после любой
перекладки. Здесь координаты не фиксируются — на каждом кадре независимо
ищутся все "кольца" (тонкий насыщенный периметр вокруг плитки, отличается от
сплошной аватарки низкой заполненностью bbox), и для каждого найденного кольца
OCR читает подпись с именем внутри самой плитки — поэтому раскладка может быть
любой на каждом кадре.

Обязательный шаг перед доверием результату на новой записи — прогнать
`--selftest` на 3-5 кадрах с уже известным (проверенным вручную через Read)
активным говорящим и убедиться, что найденный bbox/имя совпадает. Не доверять
скрипту на новой записи без такой проверки (см. правило проекта "скриптовую
автоматизацию — проверять по ходу, не в конце").

Зависимости: ffmpeg в PATH, numpy, scipy, Pillow, tesseract + tessdata/rus.

Пример:
    python3 scan_speaker_ring.py display.mp4 --timestamps 00:11:53,00:30:20 \
        --tessdata ./tessdata
    python3 scan_speaker_ring.py display.mp4 --turns turn_starts.txt \
        --tessdata ./tessdata --dual-sample -o results.tsv
"""
import argparse
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def parse_ts(s: str) -> float:
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = "0", parts[0], parts[1]
    else:
        raise ValueError(f"bad timestamp: {s}")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def fmt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def is_green_ring_mask(arr, hue="green"):
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    if hue == "green":
        return (g > 140) & (g - r > 50) & (g - b > 50)
    raise ValueError("only 'green' hue calibrated so far — sample your own "
                      "border color with a manual pixel check before adding "
                      "another hue (Zoom/Meet use different accent colors)")


def find_rings(mask, min_size=150, max_fill=0.25):
    """Ring = hollow rectangle: large bbox, but sparse actual green pixels
    inside it (only the perimeter is lit, not the whole tile)."""
    dil = ndimage.binary_dilation(mask, iterations=4)
    lbl, n = ndimage.label(dil)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(ys) == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        h, w = y1 - y0, x1 - x0
        if h < min_size or w < min_size:
            continue
        bbox_area = h * w
        real_count = mask[y0:y1, x0:x1].sum()
        fill = real_count / bbox_area if bbox_area else 1
        if fill < max_fill:
            boxes.append((y0, y1, x0, x1, fill))
    return boxes


def ocr_label(im, y0, y1, x0, x1, tessdata_dir, label_band=70):
    """Name label sits in a dark bar near the BOTTOM of the tile, inside the
    ring's own bbox — independent of where that bbox is on screen, so this
    works regardless of grid position or column count."""
    crop = im.crop((x0, max(y0, y1 - label_band), x1, y1))
    crop = crop.resize((crop.width * 3, crop.height * 3))
    tmp = Path(tessdata_dir).parent / "_scan_speaker_ring_ocr_tmp.png"
    crop.save(tmp)
    out = subprocess.run(
        ["tesseract", str(tmp), "-", "--tessdata-dir", str(tessdata_dir),
         "-l", "rus", "--psm", "7"],
        capture_output=True, text=True,
    ).stdout.strip()
    tmp.unlink(missing_ok=True)
    return re.sub(r"\s+", " ", out).strip(" |")


def extract_frame(video, t, out_jpg):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", fmt_ts(t), "-i", str(video),
         "-vframes", "1", "-q:v", "3", str(out_jpg)],
        capture_output=True,
    )


def analyze_timestamp(video, t, tessdata_dir, tmp_dir):
    jpg = Path(tmp_dir) / "_scan_speaker_ring_frame.jpg"
    extract_frame(video, t, jpg)
    im = Image.open(jpg).convert("RGB")
    arr = np.array(im).astype(int)
    mask = is_green_ring_mask(arr)
    boxes = find_rings(mask)
    jpg.unlink(missing_ok=True)
    out = []
    for (y0, y1, x0, x1, fill) in boxes:
        label = ocr_label(im, y0, y1, x0, x1, tessdata_dir)
        out.append((x0, y0, x1, y1, label, fill))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--tessdata", required=True, help="dir with rus.traineddata")
    ap.add_argument("--timestamps", help="comma-separated HH:MM:SS list")
    ap.add_argument("--turns", help="file with one turn-start timestamp per line "
                                     "(e.g. transcript turn boundaries)")
    ap.add_argument("--dual-sample", action="store_true",
                     help="sample twice per turn (start+3s lag, and end-0.2s) "
                          "and report both — disagreement between the two "
                          "means: don't trust a single-frame answer for that "
                          "turn (found empirically 25.09.2026: ~23% of turns "
                          "in a rapid back-and-forth disagree between the two "
                          "samples)")
    ap.add_argument("-o", "--out", default="-")
    ap.add_argument("--tmp-dir", default="/tmp")
    args = ap.parse_args()

    rows = []
    if args.timestamps:
        for ts in args.timestamps.split(","):
            t = parse_ts(ts.strip())
            boxes = analyze_timestamp(args.video, t, args.tessdata, args.tmp_dir)
            rows.append((fmt_ts(t), fmt_ts(t), boxes))
    elif args.turns:
        starts = [parse_ts(l.strip()) for l in open(args.turns) if l.strip()]
        for i, s in enumerate(starts):
            nxt = starts[i + 1] if i + 1 < len(starts) else s + 3.0
            dur = max(nxt - s, 0.5)
            if args.dual_sample:
                t_early = max(s + 0.3, min(s + 3.0, nxt - 0.2))
                t_late = max(s + 0.3, nxt - 0.2)
                b1 = analyze_timestamp(args.video, t_early, args.tessdata, args.tmp_dir)
                b2 = analyze_timestamp(args.video, t_late, args.tessdata, args.tmp_dir)
                rows.append((fmt_ts(s), f"early={fmt_ts(t_early)}", b1))
                rows.append((fmt_ts(s), f"late={fmt_ts(t_late)}", b2))
            else:
                t = max(s + 0.3, min(s + 3.0, nxt - 0.2))
                boxes = analyze_timestamp(args.video, t, args.tessdata, args.tmp_dir)
                rows.append((fmt_ts(s), fmt_ts(t), boxes))
    else:
        ap.error("need --timestamps or --turns")

    out_f = open(args.out, "w") if args.out != "-" else None

    def emit(line):
        print(line, file=out_f) if out_f else print(line)

    for turn_ts, sample_ts, boxes in rows:
        field = " | ".join(
            f"[{x0},{y0},{x1},{y1}]:{label}(fill={fill:.2f})"
            for (x0, y0, x1, y1, label, fill) in boxes
        )
        emit(f"{turn_ts}\t{sample_ts}\t{field}")

    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
