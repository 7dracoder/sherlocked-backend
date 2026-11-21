"""Real waveform prosody extraction and word-timestamped speech recognition."""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Sequence

import numpy as np

from .schemas import UtteranceSegment, WordTimestamp

_WHISPER_MODEL_CACHE: dict[str, object] = {}


def extract_mono_wav_16k(media_path: Path, out_wav: Path | None = None) -> Path:
    """Extracts 16kHz mono 16-bit PCM WAV from any audio or video file via ffmpeg."""
    if out_wav is None:
        fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        fd.close()
        out_wav = Path(fd.name)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(out_wav),
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size < 100:
        raise RuntimeError(f"Failed to extract 16kHz WAV from {media_path}")
    return out_wav


def load_pcm_float32(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sampwidth == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return arr, sr


def compute_segment_prosody(
    samples: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    prev_end_sec: float = 0.0,
) -> dict[str, float | bool]:
    i0 = max(0, int(start_sec * sr))
    i1 = min(len(samples), int(end_sec * sr))
    seg = samples[i0:i1]
    if seg.size < int(0.05 * sr):
        return {
            "rms_energy": 0.0,
            "mean_pitch_hz": 0.0,
            "pitch_std_hz": 0.0,
            "pause_before_sec": max(0.0, start_sec - prev_end_sec),
            "overlapping_energy": False,
        }

    rms = float(np.sqrt(np.mean(seg * seg)))

    # Framewise autocorrelation F0 estimation (75 Hz - 380 Hz)
    win = int(0.032 * sr)
    hop = int(0.016 * sr)
    min_lag = max(1, int(sr / 380.0))
    max_lag = max(min_lag + 2, int(sr / 75.0))
    pitches: list[float] = []
    peak_periodicities: list[float] = []

    for pos in range(0, max(1, len(seg) - win), hop):
        frame = seg[pos : pos + win]
        f_rms = float(np.sqrt(np.mean(frame * frame)))
        if f_rms < max(0.025, 0.32 * rms):
            continue
        frame = frame - float(np.mean(frame))
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        if corr[0] <= 1e-8:
            continue
        norm_corr = corr / corr[0]
        window_slice = norm_corr[min_lag : min(len(norm_corr), max_lag)]
        if window_slice.size < 3:
            continue
        peak_idx = int(np.argmax(window_slice))
        peak_val = float(window_slice[peak_idx])
        peak_periodicities.append(peak_val)
        if peak_val >= 0.32:
            lag = min_lag + peak_idx
            pitches.append(float(sr / lag))

    mean_f0 = float(np.mean(pitches)) if pitches else 0.0
    std_f0 = float(np.std(pitches)) if len(pitches) > 1 else 0.0
    # Two simultaneous talkers degrade single-F0 autocorrelation periodicity (<0.60) despite high speech energy
    overlap_detected = (
        len(peak_periodicities) >= 20
        and rms > 0.045
        and float(np.median(peak_periodicities)) < 0.595
    )

    # Measure true acoustic silence gap before start_sec
    gap_i0 = max(0, int(prev_end_sec * sr))
    gap_seg = samples[gap_i0:i0]
    pause_sec = max(0.0, start_sec - prev_end_sec)
    if gap_seg.size > int(0.04 * sr):
        # Refine onset if leading silence sits inside start_sec..end_sec
        env_win = max(1, int(0.02 * sr))
        for k in range(0, min(len(seg) - env_win, int(0.8 * sr)), env_win):
            if float(np.sqrt(np.mean(seg[k : k + env_win] ** 2))) > max(0.015, 0.22 * rms):
                pause_sec += k / float(sr)
                break

    return {
        "rms_energy": round(rms, 5),
        "mean_pitch_hz": round(mean_f0, 2),
        "pitch_std_hz": round(std_f0, 2),
        "pause_before_sec": round(pause_sec, 3),
        "overlapping_energy": bool(overlap_detected),
    }


def enrich_utterances_with_audio(
    utterances: Sequence[UtteranceSegment],
    wav_path: Path,
) -> list[UtteranceSegment]:
    samples, sr = load_pcm_float32(wav_path)
    enriched: list[UtteranceSegment] = []
    prev_end = 0.0
    for u in utterances:
        stats = compute_segment_prosody(samples, sr, u.start_sec, u.end_sec, prev_end)
        dur = max(0.05, u.end_sec - u.start_sec)
        word_count = len([w for w in u.text.split() if w.strip()])
        wps = round(word_count / dur, 2)
        updated = u.model_copy(
            update={
                "rms_energy": float(stats["rms_energy"]),
                "mean_pitch_hz": float(stats["mean_pitch_hz"]),
                "pitch_std_hz": float(stats["pitch_std_hz"]),
                "pause_before_sec": float(stats["pause_before_sec"]),
                "speech_rate_wps": wps,
                "overlapping_speech": bool(u.overlapping_speech or stats["overlapping_energy"]),
            }
        )
        enriched.append(updated)
        prev_end = max(prev_end, u.end_sec)
    return enriched


def transcribe_media_with_whisper(
    media_path: Path,
    model_size: str = "base.en",
) -> list[UtteranceSegment]:
    """Runs real faster-whisper speech-to-text in an isolated worker (preventing PyAV/OpenCV dylib clash)."""
    import hashlib
    import json
    import sys
    from .config import ROOT_DIR

    tmp_wav: Path | None = None
    try:
        if media_path.suffix.lower() == ".wav":
            wav_path = media_path
        else:
            tmp_wav = extract_mono_wav_16k(media_path)
            wav_path = tmp_wav

        wav_sha = hashlib.sha256(wav_path.read_bytes()).hexdigest()[:24]
        cache_dir = ROOT_DIR / "backend" / ".asr_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{model_size}_{wav_sha}.json"

        if cache_file.exists():
            raw_segments = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            proc = subprocess.run(
                [sys.executable, "-m", "backend._whisper_worker", str(wav_path), model_size],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                check=True,
            )
            raw_segments = json.loads(proc.stdout)
            cache_file.write_text(json.dumps(raw_segments), encoding="utf-8")

        utterances = [UtteranceSegment.model_validate(item) for item in raw_segments]
        return enrich_utterances_with_audio(utterances, wav_path)
    finally:
        if tmp_wav is not None:
            tmp_wav.unlink(missing_ok=True)
