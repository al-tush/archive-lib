#!/usr/bin/env python3
import sys
import json
from pathlib import Path

import sherpa_onnx
import soundfile as sf

MODELS = Path(__file__).parent

def init_speaker_diarization(num_speakers=2, cluster_threshold=0.5):
    segmentation_model = str(MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx")
    embedding_extractor_model = str(MODELS / "wespeaker_en_voxceleb_resnet34_LM.onnx")

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation_model
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding_extractor_model
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=cluster_threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("config invalid — check model paths")
    return sherpa_onnx.OfflineSpeakerDiarization(config)


def main():
    wav_path = sys.argv[1]
    out_json = sys.argv[2]
    num_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio[:, 0]

    sd = init_speaker_diarization(num_speakers=num_speakers)
    if sample_rate != sd.sample_rate:
        raise RuntimeError(f"expected {sd.sample_rate}Hz, got {sample_rate}Hz — resample first")

    def progress(done, total):
        print(f"progress: {done/total*100:.1f}%", file=sys.stderr)
        return 0

    result = sd.process(audio, callback=progress).sort_by_start_time()

    segments = [{"start": r.start, "end": r.end, "speaker": r.speaker} for r in result]
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(segments)} segments to {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
