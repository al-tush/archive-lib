#!/usr/bin/env python3
"""
Детекция переключения "подсвеченного говорящего" по видеоряду звонка.

Многие видеоконференции (Yandex Telemost и т.п.) подсвечивают плитку активного
говорящего тонкой рамкой насыщенного цвета. Это независимый от аудио сигнал,
которым можно проверять/поправлять границы аудио-диаризации на спорных
участках — без ручного покадрового разбора видео человеком или ИИ.

Два режима:

  calibrate  — по нескольким кадрам видео находит прямоугольные "рамки-кольца"
               (тонкий насыщенный периметр вокруг плитки на тёмном/сером фоне),
               кластеризует их в позиции плиток, пишет calibration.json.
               Проверить/дополнить его руками (добавить "label" — имя человека
               на плитке) перед scan.

  scan       — по calibration.json прогоняет видео целиком с заданным fps,
               для каждой плитки считает по узкой полосе вдоль её периметра
               долю пикселей, совпадающих с калиброванным цветом рамки, с
               гистерезисом/дебаунсом схлопывает в интервалы, сдвигает на лаг
               рендера (--lag) и пишет таймлайн (JSON + txt).

Зависимости: ffmpeg/ffprobe в PATH, numpy, scipy, Pillow (PIL — только в
calibrate, для сохранения кадра предпросмотра).

Пример:
    python3 video_speaker_border.py calibrate call.mp4 -o calibration.json
    # руками проверить/подписать calibration.json (поле "label" у каждой плитки)
    python3 video_speaker_border.py scan call.mp4 calibration.json -o timeline.json --lag 3.0
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TESSDATA_DIR = SCRIPT_DIR / "tessdata"
DEFAULT_ALIASES_PATH = SCRIPT_DIR / "speaker_aliases.json"


def ffprobe_dims(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)["streams"][0]
    return int(info["width"]), int(info["height"]), float(info.get("duration") or 0.0)


def frame_source(video, fps, scale_w=None, scale_h=None, start=0.0):
    """Генератор (timestamp_s, ndarray[H,W,3] uint8) — декодирует один раз, стримом.
    `start` (сек) — быстрый seek (-ss до -i) для отладки на куске записи, а не
    на всём файле; временные метки в выдаче — абсолютные (start + i/fps)."""
    orig_w, orig_h, duration = ffprobe_dims(video)
    vf = f"fps={fps}"
    w, h = orig_w, orig_h
    if scale_w:
        scale_h = scale_h or round(orig_h * scale_w / orig_w)
        vf += f",scale={scale_w}:{scale_h}"
        w, h = scale_w, scale_h
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video), "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes = w * h * 3
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
            yield start + idx / fps, frame
            idx += 1
    finally:
        proc.stdout.close()
        proc.wait()
    return w, h, duration


def saturated_mask(frame, sat_min=60, val_min=180, dominance_min=60):
    """Пиксели, где один канал явно доминирует над двумя другими и достаточно ярок —
    ловит акцентные UI-цвета (зелёный/синий/оранжевый периметр), не завязано на
    конкретный цвет платформы."""
    r = frame[:, :, 0].astype(int)
    g = frame[:, :, 1].astype(int)
    b = frame[:, :, 2].astype(int)
    channels = [r, g, b]
    mask = np.zeros(frame.shape[:2], dtype=bool)
    for i in range(3):
        dom = channels[i]
        others = [channels[j] for j in range(3) if j != i]
        cond = (dom > val_min) & (dom - others[0] > dominance_min) & (dom - others[1] > dominance_min)
        mask |= cond
    return mask


def ring_components(mask, frame_area, min_area_frac=0.03, max_fill_ratio=0.12, min_edge_cover=0.6):
    """Находит компоненты маски, форма которых — тонкий прямоугольный периметр
    ("кольцо"): большой bbox, но мало закрашенных пикселей внутри него, пиксели
    есть у всех 4 краёв bbox (не дуга/угол, а полный контур)."""
    lbl, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return []
    results = []
    objs = ndimage.find_objects(lbl)
    for comp_id, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = sl
        comp_mask = (lbl[sl] == comp_id)
        size = comp_mask.sum()
        bbox_area = comp_mask.shape[0] * comp_mask.shape[1]
        if bbox_area < frame_area * min_area_frac:
            continue
        fill_ratio = size / bbox_area
        if fill_ratio > max_fill_ratio:
            continue
        # покрытие каждого из 4 краёв bbox (периметр должен идти по всем сторонам)
        top_cov = comp_mask[0, :].mean()
        bot_cov = comp_mask[-1, :].mean()
        left_cov = comp_mask[:, 0].mean()
        right_cov = comp_mask[:, -1].mean()
        if min(top_cov, bot_cov, left_cov, right_cov) < min_edge_cover:
            continue
        y0, y1 = ys.start, ys.stop
        x0, x1 = xs.start, xs.stop
        results.append({
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "size": int(size),
            "fill_ratio": float(fill_ratio),
        })
    return results


def median_color(frame, bbox, mask):
    x0, y0, x1, y1 = bbox
    sub_frame = frame[y0:y1, x0:x1]
    sub_mask = mask[y0:y1, x0:x1]
    pixels = sub_frame[sub_mask]
    if len(pixels) == 0:
        return [0, 0, 0]
    return [int(v) for v in np.median(pixels, axis=0)]


def cluster_boxes(candidates, tol=30):
    """Группирует bbox-кандидаты по позиции (близкие центры/размеры — одна плитка)."""
    clusters = []  # each: {"boxes": [...], "colors": [...]}
    for cand in candidates:
        x0, y0, x1, y1 = cand["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        placed = False
        for cl in clusters:
            bx0, by0, bx1, by1 = cl["ref_bbox"]
            bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
            if abs(cx - bcx) < tol and abs(cy - bcy) < tol:
                cl["boxes"].append(cand["bbox"])
                cl["colors"].append(cand["color"])
                placed = True
                break
        if not placed:
            clusters.append({"ref_bbox": cand["bbox"], "boxes": [cand["bbox"]], "colors": [cand["color"]]})
    tiles = []
    for cl in clusters:
        boxes = np.array(cl["boxes"])
        colors = np.array(cl["colors"])
        tiles.append({
            "bbox": [int(v) for v in np.median(boxes, axis=0)],
            "color": [int(v) for v in np.median(colors, axis=0)],
            "n_observations": len(cl["boxes"]),
        })
    return tiles


_LABEL_CLEAN_RE = re.compile(r"[^A-Za-zА-Яа-яЁё0-9 .\-]")


def normalize_label(text):
    """Чистит OCR-мусор (иконка микрофона, обрывки рамки на границе кадра) —
    оставляет только буквы/цифры/пробел/точку/дефис, схлопывает пробелы."""
    text = _LABEL_CLEAN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ocr_label(frame, band_bbox, tessdata_dir, lang="rus+eng", upscale=2):
    """OCR узкой полосы кадра (имя участника под плиткой). Отдельный процесс
    tesseract на кадр — приемлемо, т.к. используется только при разреженной
    выборке (calibrate), не на каждом кадре сканирования."""
    x0, y0, x1, y1 = band_bbox
    h, w = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 10 or y1 - y0 < 3:
        return ""
    crop = Image.fromarray(frame[y0:y1, x0:x1])
    if upscale != 1:
        crop = crop.resize((crop.width * upscale, crop.height * upscale))
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        crop.save(tmp.name)
        env = dict(os.environ, TESSDATA_PREFIX=str(tessdata_dir))
        try:
            out = subprocess.run(
                ["tesseract", tmp.name, "-", "--psm", "7", "-l", lang],
                capture_output=True, env=env, timeout=15,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ""
    stdout_text = out.stdout.decode("utf-8", errors="replace")
    return normalize_label(stdout_text)


def label_band_for_tile(bbox, top_frac=0.90, bottom_frac=0.985):
    """Полоса с именем участника — нижняя часть плитки (под аватаркой), где
    Telemost/аналоги рисуют плашку с именем. Доли по умолчанию откалиброваны
    на записи Yandex Telemost 3840x2560 (см. call-logs/2026-08-20-...);
    для другого UI/разрешения могут потребовать подстройки через CLI."""
    x0, y0, x1, y1 = bbox
    h = y1 - y0
    return [x0, int(y0 + top_frac * h), x1, int(y0 + bottom_frac * h)]


def sample_labels(video, tiles, samples, tessdata_dir):
    """Сэмплирует кадры по всей длине видео, OCR-ит полосу с именем под каждой
    плиткой. Не предполагает, что звонок идёт с начала записи (запись экрана
    может стартовать раньше) — просто собирает частоты текста по каждому
    слоту и позже берёт то, что реально встречалось чаще всего; кадры до
    начала звонка или с посторонним содержимым дают пустой/мусорный OCR и
    тонут в счётчике на фоне повторяющегося настоящего лейбла."""
    orig_w, orig_h, duration = ffprobe_dims(video)
    n = samples
    timestamps = [duration * (i + 0.5) / n for i in range(n)] if duration else [0]
    per_tile = {t["id"]: [] for t in tiles}
    for t in timestamps:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(video),
             "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            capture_output=True, check=True,
        ).stdout
        if len(out) < orig_w * orig_h * 3:
            continue
        frame = np.frombuffer(out, dtype=np.uint8).reshape((orig_h, orig_w, 3))
        for tile in tiles:
            band = label_band_for_tile(tile["bbox"])
            text = ocr_label(frame, band, tessdata_dir)
            if text:
                per_tile[tile["id"]].append((t, text))
    return per_tile


def load_aliases(path):
    path = Path(path)
    if not path.exists():
        return {"aliases": {}}
    return json.loads(path.read_text())


def save_aliases(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2))


def apply_labels(tiles, raw_by_tile, aliases_path):
    """Для каждой плитки — статистика встреченных сырых лейблов (частоты,
    первое/последнее появление), автовыбор самого частого как основного,
    сопоставление через словарь алиасов. Новые нераспознанные лейблы
    дописываются в словарь с canonical=null — файл растёт сам, дозаполнение
    (кто такой 'DK') остаётся за человеком."""
    aliases_data = load_aliases(aliases_path)
    aliases = aliases_data.setdefault("aliases", {})
    unresolved = []
    for tile in tiles:
        observations = raw_by_tile.get(tile["id"], [])
        counts = Counter(text for _, text in observations)
        stats = []
        for text, count in counts.most_common():
            times = [ts for ts, t in observations if t == text]
            stats.append({
                "text": text,
                "count": count,
                "first_seen": round(min(times), 1),
                "last_seen": round(max(times), 1),
            })
        tile["raw_labels"] = stats
        if stats:
            top_text = stats[0]["text"]
            if top_text in aliases:
                canonical = aliases[top_text]
                tile["label"] = canonical if canonical else top_text
                if not canonical:
                    unresolved.append(top_text)
            else:
                aliases[top_text] = None
                tile["label"] = top_text
                unresolved.append(top_text)
        else:
            tile["label"] = None
    save_aliases(aliases_path, aliases_data)
    return unresolved


def cmd_calibrate(args):
    video = Path(args.video)
    orig_w, orig_h, duration = ffprobe_dims(video)
    n = args.samples
    timestamps = [duration * (i + 0.5) / n for i in range(n)] if duration else [0]
    frame_area = orig_w * orig_h
    candidates = []
    for t in timestamps:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(video),
             "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
            capture_output=True, check=True,
        ).stdout
        if len(out) < orig_w * orig_h * 3:
            continue
        frame = np.frombuffer(out, dtype=np.uint8).reshape((orig_h, orig_w, 3))
        mask = saturated_mask(frame)
        for comp in ring_components(mask, frame_area):
            x0, y0, x1, y1 = comp["bbox"]
            comp["color"] = median_color(frame, comp["bbox"], mask)
            candidates.append(comp)

    tiles = cluster_boxes(candidates, tol=args.cluster_tol)
    tiles = [t for t in tiles if t["n_observations"] >= args.min_observations]
    tiles.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    for i, t in enumerate(tiles):
        t["id"] = i
        t["label"] = None

    unresolved = []
    if tiles and not args.no_labels:
        raw_by_tile = sample_labels(video, tiles, args.label_samples, Path(args.tessdata_dir))
        unresolved = apply_labels(tiles, raw_by_tile, Path(args.aliases))

    calib = {
        "video": str(video),
        "frame_w": orig_w,
        "frame_h": orig_h,
        "samples": n,
        "tiles": tiles,
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(calib, ensure_ascii=False, indent=2))
    print(f"Найдено плиток: {len(tiles)} (из {len(candidates)} кандидатов на {n} кадрах)")
    for t in tiles:
        top = t.get("raw_labels", [{}])[0].get("text") if t.get("raw_labels") else None
        print(f"  id={t['id']} bbox={t['bbox']} color={t['color']} n_obs={t['n_observations']} "
              f"label={t['label']!r} (raw={top!r})")
    print(f"Записано: {out_path}")
    if unresolved:
        print(f"Нераспознанные лейблы (добавлены в {args.aliases} с canonical=null, "
              f"дозаполнить руками): {sorted(set(unresolved))}", file=sys.stderr)
    if not tiles:
        print("Ничего не найдено — либо в видео нет UI с рамкой активного говорящего, "
              "либо нужно ослабить пороги (--min-area-frac и т.п. пока не вынесены "
              "в CLI, править в saturated_mask/ring_components).", file=sys.stderr)


def cmd_relabel(args):
    """Переприменить словарь алиасов к уже собранной calibration.json — без
    повторного OCR/декодирования видео. Использовать после того, как руками
    дозаполнили speaker_aliases.json (canonical вместо null)."""
    calib_path = Path(args.calibration)
    calib = json.loads(calib_path.read_text())
    aliases_data = load_aliases(args.aliases)
    aliases = aliases_data.setdefault("aliases", {})
    for tile in calib["tiles"]:
        stats = tile.get("raw_labels") or []
        if not stats:
            continue
        top_text = stats[0]["text"]
        if top_text not in aliases:
            aliases[top_text] = None
        canonical = aliases[top_text]
        tile["label"] = canonical if canonical else top_text
    save_aliases(args.aliases, aliases_data)
    calib_path.write_text(json.dumps(calib, ensure_ascii=False, indent=2))
    for t in calib["tiles"]:
        print(f"  id={t['id']} label={t['label']!r}")
    print(f"Обновлено: {calib_path}")


def strip_indices(bbox, strip):
    """Индексы узкой полосы по периметру bbox (для scan — не весь кадр, а только край плитки)."""
    x0, y0, x1, y1 = bbox
    return {
        "top": (slice(y0, min(y0 + strip, y1)), slice(x0, x1)),
        "bottom": (slice(max(y1 - strip, y0), y1), slice(x0, x1)),
        "left": (slice(y0, y1), slice(x0, min(x0 + strip, x1))),
        "right": (slice(y0, y1), slice(max(x1 - strip, x0), x1)),
    }


def tile_active_score(frame, bbox, color, strip, color_tol):
    color = np.array(color)
    sides = strip_indices(bbox, strip)
    total = 0
    match = 0
    for sl in sides.values():
        sub = frame[sl[0], sl[1]].astype(int)
        dist = np.abs(sub - color).sum(axis=-1)
        match += int((dist < color_tol).sum())
        total += sub.shape[0] * sub.shape[1]
    return match / total if total else 0.0


def merge_segments(states, timestamps, min_on, min_off):
    """states: список bool по кадрам (сырое покадровое решение по порогу).
    Возвращает список (start_ts, end_ts) с гистерезисом по min_on/min_off кадров."""
    segments = []
    confirmed = False
    run_true = 0
    run_false = 0
    seg_start = None
    for i, s in enumerate(states):
        if s:
            run_true += 1
            run_false = 0
        else:
            run_false += 1
            run_true = 0
        if not confirmed and run_true >= min_on:
            confirmed = True
            seg_start = timestamps[i - min_on + 1]
        elif confirmed and run_false >= min_off:
            confirmed = False
            seg_end = timestamps[i - min_off + 1]
            segments.append((seg_start, seg_end))
            seg_start = None
    if confirmed and seg_start is not None:
        segments.append((seg_start, timestamps[-1]))
    return segments


def cmd_scan(args):
    video = Path(args.video)
    calib = json.loads(Path(args.calibration).read_text())
    tiles = calib["tiles"]
    calib_w = calib["frame_w"]

    scale_w = args.scale
    scale_factor = (scale_w / calib_w) if scale_w else 1.0
    strip = max(2, round(args.strip * scale_factor))

    scaled_tiles = []
    for t in tiles:
        x0, y0, x1, y1 = t["bbox"]
        scaled_tiles.append({
            **t,
            "bbox_scaled": [int(v * scale_factor) for v in (x0, y0, x1, y1)],
        })

    per_tile_states = {t["id"]: [] for t in scaled_tiles}
    timestamps = []
    n_frames = 0
    for ts, frame in frame_source(video, args.fps, scale_w=scale_w, start=args.start):
        timestamps.append(ts)
        for t in scaled_tiles:
            score = tile_active_score(frame, t["bbox_scaled"], t["color"], strip, args.color_tol)
            per_tile_states[t["id"]].append(score >= args.threshold)
        n_frames += 1
        if args.max_frames and n_frames >= args.max_frames:
            break

    min_on = max(1, round(args.min_on_s * args.fps))
    min_off = max(1, round(args.min_off_s * args.fps))

    timeline = []
    for t in scaled_tiles:
        segs = merge_segments(per_tile_states[t["id"]], timestamps, min_on, min_off)
        for start, end in segs:
            timeline.append({
                "tile_id": t["id"],
                "label": t.get("label") or f"tile-{t['id']}",
                "start": round(max(0.0, start - args.lag), 2),
                "end": round(max(0.0, end - args.lag), 2),
            })
    timeline.sort(key=lambda s: s["start"])

    out_path = Path(args.output)
    out_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2))

    txt_path = out_path.with_suffix(".txt")
    lines = []
    for seg in timeline:
        s, e = seg["start"], seg["end"]
        lines.append(f"[{int(s//60):02d}:{s%60:05.2f}-{int(e//60):02d}:{e%60:05.2f}] {seg['label']}")
    txt_path.write_text("\n".join(lines))

    print(f"Кадров обработано: {n_frames}, интервалов: {len(timeline)}")
    print(f"Записано: {out_path} и {txt_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="найти плитки-рамки в видео")
    c.add_argument("video")
    c.add_argument("-o", "--output", default="calibration.json")
    c.add_argument("--samples", type=int, default=25, help="сколько кадров сэмплировать по всей длине видео")
    c.add_argument("--cluster-tol", type=int, default=30, help="допуск (px) для группировки bbox в одну плитку")
    c.add_argument("--min-observations", type=int, default=2, help="минимум наблюдений кластера, чтобы считать его плиткой")
    c.add_argument("--no-labels", action="store_true", help="не делать OCR имён под плитками (только геометрия/цвет)")
    c.add_argument("--label-samples", type=int, default=20, help="сколько кадров сэмплировать для OCR имён (независимо от --samples для геометрии)")
    c.add_argument("--tessdata-dir", default=str(DEFAULT_TESSDATA_DIR), help="каталог с *.traineddata для tesseract")
    c.add_argument("--aliases", default=str(DEFAULT_ALIASES_PATH), help="JSON-словарь 'сырой лейбл' -> каноническое имя (пополняется автоматически)")
    c.set_defaults(func=cmd_calibrate)

    r = sub.add_parser("relabel", help="переприменить speaker_aliases.json к готовому calibration.json без повторного OCR")
    r.add_argument("calibration")
    r.add_argument("--aliases", default=str(DEFAULT_ALIASES_PATH))
    r.set_defaults(func=cmd_relabel)

    s = sub.add_parser("scan", help="прогнать видео целиком по calibration.json")
    s.add_argument("video")
    s.add_argument("calibration")
    s.add_argument("-o", "--output", default="timeline.json")
    s.add_argument("--fps", type=float, default=5.0)
    s.add_argument("--scale", type=int, default=None, help="ширина кадра для скана (по умолчанию — родное разрешение); высота считается пропорционально")
    s.add_argument("--strip", type=int, default=20, help="ширина полосы (px) вдоль периметра плитки для проверки цвета")
    s.add_argument("--color-tol", type=int, default=90, help="допуск суммы |ΔR|+|ΔG|+|ΔB| для совпадения с цветом рамки")
    s.add_argument("--threshold", type=float, default=0.22, help="доля пикселей полосы, совпавших с цветом, чтобы считать кадр 'активным' (полоса тонкая, ~0.2-0.3 — уже уверенное совпадение, не 0.5+)")
    s.add_argument("--min-on-s", type=float, default=0.4, help="рамка должна держаться не меньше стольки секунд, чтобы засчитать включение")
    s.add_argument("--min-off-s", type=float, default=0.4, help="столько секунд без рамки, чтобы засчитать выключение")
    s.add_argument("--lag", type=float, default=0.0, help="известная задержка рендера рамки относительно реальной речи (секунды) — вычитается из таймингов")
    s.add_argument("--max-frames", type=int, default=None, help="ограничить число кадров (для отладки)")
    s.add_argument("--start", type=float, default=0.0, help="начать с этой секунды видео (быстрый seek, для отладки на куске записи)")
    s.set_defaults(func=cmd_scan)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
