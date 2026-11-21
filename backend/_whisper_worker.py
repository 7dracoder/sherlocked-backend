"""Isolated process worker for faster-whisper to avoid PyAV + OpenCV libavdevice Objective-C collisions."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


def main() -> None:
    wav_path = Path(sys.argv[1])
    model_size = sys.argv[2] if len(sys.argv) > 2 else "base.en"

    with wave.open(str(wav_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
        n_channels = wf.getnchannels()
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments_iter, _ = model.transcribe(
        pcm,
        word_timestamps=True,
        vad_filter=True,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )

    out_segments = []
    for idx, seg in enumerate(segments_iter):
        text = seg.text.strip()
        if not text:
            continue
        words = []
        for w in seg.words or []:
            wt = w.word.strip()
            if wt:
                words.append(
                    {
                        "word": wt,
                        "start_sec": round(float(w.start), 3),
                        "end_sec": round(float(w.end), 3),
                        "confidence": round(float(getattr(w, "probability", 0.9)), 3),
                    }
                )
        out_segments.append(
            {
                "segment_id": f"utt_{idx+1:03d}",
                "speaker_id": "speaker_primary",
                "text": text,
                "start_sec": round(float(seg.start), 3),
                "end_sec": round(float(seg.end), 3),
                "words": words,
            }
        )

    sys.stdout.write(json.dumps(out_segments))


if __name__ == "__main__":
    main()
