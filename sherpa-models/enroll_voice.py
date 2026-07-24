#!/usr/bin/env python3
"""
Enroll a named speaker's voice from a wav segment (or several segments) into
a local voice-profile store, for later automatic speaker labeling of
diarization output.

Usage:
  enroll_voice.py add <name> <wav_file> [start end]   # add one embedding
  enroll_voice.py identify <wav_file> <diarization.json>   # label clusters
"""
import sys
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import sherpa_onnx

MODELS = Path(__file__).parent
PROFILES_PATH = MODELS / "voice_profiles.json"
EMBEDDING_MODEL = str(MODELS / "wespeaker_en_voxceleb_resnet34_LM.onnx")


def load_profiles():
    if PROFILES_PATH.exists():
        with open(PROFILES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_profiles(profiles):
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)


def make_extractor():
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMBEDDING_MODEL)
    return sherpa_onnx.SpeakerEmbeddingExtractor(config)


def read_wav_segment(wav_path, start=None, end=None):
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio[:, 0]
    if start is not None:
        audio = audio[int(start * sample_rate):int(end * sample_rate)]
    return audio, sample_rate


def compute_embedding(extractor, audio, sample_rate):
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sample_rate, waveform=audio)
    stream.input_finished()
    return extractor.compute(stream)


def cmd_add(name, wav_file, start=None, end=None):
    extractor = make_extractor()
    audio, sr = read_wav_segment(wav_file, start, end)
    emb = compute_embedding(extractor, audio, sr)
    profiles = load_profiles()
    profiles.setdefault(name, []).append(list(map(float, emb)))
    save_profiles(profiles)
    print(f"added embedding for '{name}' ({len(profiles[name])} total samples)")


def cmd_identify(wav_file, diar_json, threshold=0.5):
    profiles = load_profiles()
    if not profiles:
        print("no voice profiles enrolled yet", file=sys.stderr)
        sys.exit(1)

    # mean-normalize each profile's embeddings into one reference vector
    refs = {}
    for name, embs in profiles.items():
        arr = np.array(embs)
        v = arr.mean(axis=0)
        refs[name] = v / np.linalg.norm(v)

    extractor = make_extractor()
    with open(diar_json, encoding="utf-8") as f:
        segments = json.load(f)

    audio, sr = read_wav_segment(wav_file)

    # group segments by speaker id, take up to 3 longest segments per speaker
    by_speaker = {}
    for s in segments:
        by_speaker.setdefault(s["speaker"], []).append(s)

    speaker_names = {}
    for spk, segs in by_speaker.items():
        segs = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)[:3]
        embs = []
        for s in segs:
            seg_audio = audio[int(s["start"] * sr):int(s["end"] * sr)]
            if len(seg_audio) < sr * 0.5:  # skip too-short clips
                continue
            embs.append(compute_embedding(extractor, seg_audio, sr))
        if not embs:
            speaker_names[spk] = f"speaker_{spk:02}"
            continue
        v = np.array(embs).mean(axis=0)
        v = v / np.linalg.norm(v)

        best_name, best_score = None, -1
        for name, ref in refs.items():
            score = float(np.dot(v, ref))
            if score > best_score:
                best_name, best_score = name, score
        if best_score >= threshold:
            speaker_names[spk] = best_name
        else:
            speaker_names[spk] = f"speaker_{spk:02}(?)"
        print(f"speaker_{spk:02} -> {speaker_names[spk]} (score={best_score:.3f})", file=sys.stderr)

    print(json.dumps(speaker_names, ensure_ascii=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "add":
        name = sys.argv[2]
        wav_file = sys.argv[3]
        start = float(sys.argv[4]) if len(sys.argv) > 4 else None
        end = float(sys.argv[5]) if len(sys.argv) > 5 else None
        cmd_add(name, wav_file, start, end)
    elif cmd == "identify":
        cmd_identify(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)
