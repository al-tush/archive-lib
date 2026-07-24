#!/usr/bin/env python3
import sys
import json
import re

def parse_srt(path):
    with open(path, encoding="utf-8") as f:
        content = f.read()
    blocks = content.strip().split("\n\n")
    segments = []
    for b in blocks:
        lines = b.split("\n")
        if len(lines) < 3:
            continue
        ts = lines[1]
        text = " ".join(lines[2:]).strip()
        m = re.match(r"(\d\d):(\d\d):(\d\d),(\d+) --> (\d\d):(\d\d):(\d\d),(\d+)", ts)
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        segments.append({"start": start, "end": end, "text": text})
    return segments


def speaker_at(diar_segments, t):
    best = None
    best_dist = None
    for d in diar_segments:
        if d["start"] <= t <= d["end"]:
            return d["speaker"]
        dist = min(abs(t - d["start"]), abs(t - d["end"]))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = d["speaker"]
    return best


def fmt_ts(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def main():
    srt_path, diar_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    srt_segments = parse_srt(srt_path)
    with open(diar_path, encoding="utf-8") as f:
        diar_segments = json.load(f)

    lines = []
    cur_speaker = None
    cur_text = []
    cur_start = None
    for seg in srt_segments:
        mid = (seg["start"] + seg["end"]) / 2
        spk = speaker_at(diar_segments, mid)
        if spk != cur_speaker:
            if cur_speaker is not None:
                lines.append(f"[{fmt_ts(cur_start)}] speaker_{cur_speaker:02}: {' '.join(cur_text)}")
            cur_speaker = spk
            cur_text = [seg["text"]]
            cur_start = seg["start"]
        else:
            cur_text.append(seg["text"])
    if cur_speaker is not None:
        lines.append(f"[{fmt_ts(cur_start)}] speaker_{cur_speaker:02}: {' '.join(cur_text)}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
    print(f"wrote {len(lines)} speaker turns to {out_path}")


if __name__ == "__main__":
    main()
