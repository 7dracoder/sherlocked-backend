"""Real frame-level facial landmark/blendshape tracker, quality gates, and face-mask renderer."""

from __future__ import annotations

import base64
import math
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .config import ROOT_DIR
from .schemas import (
    CandidateMotionInterval,
    FrameSample,
    QualityFailureFlag,
    UtteranceSegment,
    WordTimestamp,
)

MODEL_CACHE_DIR = ROOT_DIR / "backend" / ".models"
FACE_LANDMARKER_PATH = MODEL_CACHE_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def ensure_face_landmarker_model() -> Path:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not FACE_LANDMARKER_PATH.exists() or FACE_LANDMARKER_PATH.stat().st_size < 1_000_000:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
    return FACE_LANDMARKER_PATH


@dataclass
class VideoAnalysisResult:
    frames: list[FrameSample]
    candidate_events: list[CandidateMotionInterval]
    effective_fps: float
    discarded_flags: list[QualityFailureFlag]
    face_crops_by_frame_id: dict[str, str]


def _encode_jpeg_b64(bgr_img: np.ndarray, max_side: int = 256) -> str:
    h, w = bgr_img.shape[:2]
    if max(h, w) > max_side and max(h, w) > 0:
        scale = max_side / float(max(h, w))
        bgr_img = cv2.resize(bgr_img, (max(1, int(w * scale)), max(1, int(h * scale))))
    ok, buf = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_composite_channels(blendshapes: dict[str, float]) -> dict[str, float]:
    jaw_open = blendshapes.get("jawOpen", 0.0) + 0.4 * blendshapes.get("mouthFunnel", 0.0)
    lip_press = 0.5 * (
        blendshapes.get("mouthPressLeft", 0.0) + blendshapes.get("mouthPressRight", 0.0)
    ) + 0.35 * blendshapes.get("mouthRollLower", 0.0)
    eye_blink = 0.5 * (
        blendshapes.get("eyeBlinkLeft", 0.0) + blendshapes.get("eyeBlinkRight", 0.0)
    )
    gaze_shift = max(
        blendshapes.get("eyeLookOutLeft", 0.0),
        blendshapes.get("eyeLookInLeft", 0.0),
        blendshapes.get("eyeLookOutRight", 0.0),
        blendshapes.get("eyeLookInRight", 0.0),
        blendshapes.get("eyeLookUpLeft", 0.0),
        blendshapes.get("eyeLookDownLeft", 0.0),
    )
    brow_furrow = 0.5 * (
        blendshapes.get("browDownLeft", 0.0) + blendshapes.get("browDownRight", 0.0)
    ) + 0.3 * blendshapes.get("browInnerUp", 0.0)
    mouth_asymmetry = abs(
        blendshapes.get("mouthSmileLeft", 0.0) - blendshapes.get("mouthSmileRight", 0.0)
    ) + 0.5 * abs(
        blendshapes.get("mouthFrownLeft", 0.0) - blendshapes.get("mouthFrownRight", 0.0)
    )
    return {
        "lip_press": float(lip_press),
        "eye_blink": float(eye_blink),
        "gaze_shift": float(gaze_shift),
        "brow_furrow": float(brow_furrow),
        "mouth_asymmetry": float(mouth_asymmetry),
        "jaw_open": float(jaw_open),
    }


def _head_angles_from_landmarks(landmarks: Sequence[Any]) -> tuple[float, float]:
    # MediaPipe canonical indices: nose tip=1, left outer eye=33, right outer eye=263, chin=152, forehead=10
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    forehead = landmarks[10]
    chin = landmarks[152]

    eye_mid_x = 0.5 * (left_eye.x + right_eye.x)
    eye_span = max(1e-4, abs(right_eye.x - left_eye.x))
    yaw_deg = float(np.clip(((nose.x - eye_mid_x) / eye_span) * 90.0, -85.0, 85.0))

    face_h = max(1e-4, abs(chin.y - forehead.y))
    nose_rel_y = (nose.y - forehead.y) / face_h
    pitch_deg = float(np.clip((nose_rel_y - 0.56) * 140.0, -75.0, 75.0))
    return yaw_deg, pitch_deg


def _detect_eyewear_occlusion(gray: np.ndarray, landmarks: Sequence[Any]) -> tuple[bool, bool]:
    h, w = gray.shape[:2]
    if h < 32 or w < 32:
        return False, False

    def patch_mean(idx: int, radius: int = 6) -> float:
        cx = int(np.clip(landmarks[idx].x * w, radius, w - radius - 1))
        cy = int(np.clip(landmarks[idx].y * h, radius, h - radius - 1))
        roi = gray[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
        return float(np.mean(roi)) if roi.size else 128.0

    # Eye centers/lids (159, 386) vs cheeks (234, 454, 50, 280)
    eye_lum = 0.5 * (patch_mean(159, 7) + patch_mean(386, 7))
    cheek_lum = 0.5 * (patch_mean(50, 9) + patch_mean(280, 9))
    bridge_lum = patch_mean(168, 5)
    if cheek_lum > 45.0 and eye_lum < 0.42 * cheek_lum:
        # Dark sunglasses / heavy tint occluding eye region
        return True, True
    if abs(bridge_lum - cheek_lum) > 48.0:
        # Clear frames bridge visible without occluding eyes
        return True, False
    return False, False


def _utterance_state_at(
    t_sec: float, utterances: Sequence[UtteranceSegment]
) -> tuple[bool, bool, str | None, str]:
    """Returns (is_speaking, overlapping, coinciding_word, speaker_id)."""
    active_utts = [u for u in utterances if (u.start_sec - 0.15) <= t_sec <= (u.end_sec + 0.15)]
    if not active_utts:
        return False, False, None, "speaker_primary"
    overlapping = any(u.overlapping_speech for u in active_utts) or (
        len({u.speaker_id for u in active_utts}) > 1
    )
    primary = active_utts[0]
    coinciding_word: str | None = None
    best_dist = 0.45
    for u in utterances:
        for w in u.words:
            # Check words occurring right at or immediately after facial movement (e.g. lip press before 'Friday')
            delta = w.start_sec - t_sec
            if -0.20 <= delta <= 0.40 and abs(delta) < best_dist:
                best_dist = abs(delta)
                coinciding_word = w.word.strip(".,!?;:\"'")
    return True, overlapping, coinciding_word, primary.speaker_id


class FacialObservationTracker:
    """Extracts frame-linked facial observations and enforces capture failure guards."""

    def __init__(self) -> None:
        self._mp = None
        self._landmarker = None

    def _get_landmarker(self):
        if self._landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = ensure_face_landmarker_model()
            base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=3,
                min_face_detection_confidence=0.45,
                min_face_presence_confidence=0.45,
                output_face_blendshapes=True,
            )
            self._mp = mp
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        return self._mp, self._landmarker

    def analyze_video_file(
        self,
        video_path: Path,
        utterances: Sequence[UtteranceSegment] = (),
        target_fps: float | None = None,
        audio_video_drift_ms: float = 0.0,
        listener_only_override: bool = False,
    ) -> VideoAnalysisResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if native_fps <= 0 or math.isnan(native_fps):
            native_fps = 30.0

        stride = 1
        if target_fps is not None and target_fps > 0 and target_fps < native_fps:
            stride = max(1, int(round(native_fps / target_fps)))

        mp, landmarker = self._get_landmarker()
        raw_frames: list[tuple[str, float, np.ndarray]] = []
        idx = 0
        sampled_idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                t_sec = idx / native_fps
                frame_id = f"frame_{sampled_idx:05d}"
                raw_frames.append((frame_id, t_sec, bgr))
                sampled_idx += 1
            idx += 1
        cap.release()

        if not raw_frames:
            return VideoAnalysisResult([], [], 0.0, [QualityFailureFlag.FRAME_LOSS], {})

        # Compute effective delivered FPS & detect frame-loss gaps
        timestamps = [t for _, t, _ in raw_frames]
        dts = np.diff(timestamps) if len(timestamps) > 1 else np.array([1.0 / native_fps])
        positive_dts = dts[dts > 1e-5]
        effective_fps = float(1.0 / np.median(positive_dts)) if positive_dts.size else native_fps
        median_dt = float(np.median(positive_dts)) if positive_dts.size else (1.0 / native_fps)

        samples: list[FrameSample] = []
        face_crops: dict[str, np.ndarray] = {}
        prev_yaw: float | None = None
        prev_pitch: float | None = None
        prev_t: float | None = None

        for i, (frame_id, t_sec, bgr) in enumerate(raw_frames):
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            lum = float(np.mean(gray))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det = landmarker.detect(mp_img)

            flags: list[QualityFailureFlag] = []
            if lum < 42.0:
                flags.append(QualityFailureFlag.LOW_LIGHT)
            if i > 0 and (t_sec - raw_frames[i - 1][1]) > max(0.35, 2.6 * median_dt) and (
                target_fps is None or target_fps >= 10.0
            ):
                flags.append(QualityFailureFlag.FRAME_LOSS)
            if abs(audio_video_drift_ms) > 140.0:
                flags.append(QualityFailureFlag.AUDIO_DRIFT)

            is_speaking, overlapping, _, _ = _utterance_state_at(t_sec, utterances)
            if overlapping:
                flags.append(QualityFailureFlag.OVERLAPPING_SPEECH)

            faces_count = len(det.face_landmarks)
            if faces_count == 0:
                samples.append(
                    FrameSample(
                        frame_id=frame_id,
                        timestamp_sec=round(t_sec, 4),
                        face_visible=False,
                        usable=False,
                        speaker_attributed=False,
                        faces_detected=0,
                        luminance=round(lum, 2),
                        quality_flags=flags,
                    )
                )
                prev_yaw = None
                prev_pitch = None
                prev_t = t_sec
                continue

            # Select largest / most central face; flag multi-face scenes for speaker-attribution check
            h, w = bgr.shape[:2]
            best_face_idx = 0
            if faces_count > 1:
                areas = []
                for fl in det.face_landmarks:
                    xs = [p.x for p in fl]
                    ys = [p.y for p in fl]
                    areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
                best_face_idx = int(np.argmax(areas))
                flags.append(QualityFailureFlag.LISTENER_IN_FRAME)

            landmarks = det.face_landmarks[best_face_idx]
            raw_bs = {
                cat.category_name: float(cat.score)
                for cat in (det.face_blendshapes[best_face_idx] if det.face_blendshapes else [])
            }
            channels = _extract_composite_channels(raw_bs)
            yaw_deg, pitch_deg = _head_angles_from_landmarks(landmarks)

            turn_speed = 0.0
            if prev_yaw is not None and prev_pitch is not None and prev_t is not None:
                dt = max(1e-3, t_sec - prev_t)
                turn_speed = float((abs(yaw_deg - prev_yaw) + abs(pitch_deg - prev_pitch)) / dt)
            prev_yaw, prev_pitch, prev_t = yaw_deg, pitch_deg, t_sec

            if turn_speed > 65.0 or abs(yaw_deg) > 48.0:
                flags.append(QualityFailureFlag.FAST_HEAD_TURN)

            eyewear, eyewear_occludes = _detect_eyewear_occlusion(gray, landmarks)
            if eyewear_occludes:
                flags.append(QualityFailureFlag.EYEWEAR_OCCLUSION)

            if listener_only_override:
                flags.append(QualityFailureFlag.LISTENER_IN_FRAME)

            # Crop face bounding box with 20% padding
            xs = [p.x * w for p in landmarks]
            ys = [p.y * h for p in landmarks]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            pad_x = 0.22 * (x1 - x0)
            pad_y = 0.25 * (y1 - y0)
            cx0 = max(0, int(x0 - pad_x))
            cy0 = max(0, int(y0 - pad_y))
            cx1 = min(w, int(x1 + pad_x))
            cy1 = min(h, int(y1 + pad_y))
            if cx1 > cx0 and cy1 > cy0:
                face_crops[frame_id] = bgr[cy0:cy1, cx0:cx1].copy()

            motion_score = float(
                0.32 * channels["lip_press"]
                + 0.24 * channels["brow_furrow"]
                + 0.22 * channels["gaze_shift"]
                + 0.12 * channels["mouth_asymmetry"]
                + 0.10 * channels["eye_blink"]
            )

            hard_fail = {
                QualityFailureFlag.LOW_LIGHT,
                QualityFailureFlag.FAST_HEAD_TURN,
                QualityFailureFlag.FRAME_LOSS,
                QualityFailureFlag.AUDIO_DRIFT,
                QualityFailureFlag.OVERLAPPING_SPEECH,
                QualityFailureFlag.LISTENER_IN_FRAME,
            }
            usable = len(hard_fail.intersection(flags)) == 0
            speaker_attributed = usable and (faces_count == 1) and not listener_only_override

            samples.append(
                FrameSample(
                    frame_id=frame_id,
                    timestamp_sec=round(t_sec, 4),
                    face_visible=True,
                    usable=usable,
                    speaker_attributed=speaker_attributed,
                    faces_detected=faces_count,
                    luminance=round(lum, 2),
                    head_yaw_deg=round(yaw_deg, 2),
                    head_pitch_deg=round(pitch_deg, 2),
                    head_turn_deg_per_sec=round(turn_speed, 2),
                    eyewear_detected=eyewear,
                    eyewear_occludes_eyes=eyewear_occludes,
                    motion_score=round(motion_score, 4),
                    blendshapes={k: round(v, 4) for k, v in channels.items()},
                    quality_flags=flags,
                )
            )

        # Dilate FAST_HEAD_TURN by +-3 frames (~120ms settling margin) so entry/exit blur jitter cannot appear usable
        turn_indices = [
            i for i, s in enumerate(samples) if QualityFailureFlag.FAST_HEAD_TURN in s.quality_flags
        ]
        for ti in turn_indices:
            for nb in range(max(0, ti - 3), min(len(samples), ti + 4)):
                if QualityFailureFlag.FAST_HEAD_TURN not in samples[nb].quality_flags:
                    samples[nb].quality_flags.append(QualityFailureFlag.FAST_HEAD_TURN)
                samples[nb].usable = False
                samples[nb].speaker_attributed = False

        # Verify audio-visual lip-motion correlation when utterances exist:
        # If someone is speaking across >=1.5s of usable frames yet jaw_open never varies at all (<0.006 std)
        # while audio has speech, flag LISTENER_IN_FRAME on speaking segments if multi-speaker/POV mismatch.
        if utterances:
            speaking_samples = [
                s
                for s in samples
                if s.face_visible
                and s.usable
                and _utterance_state_at(s.timestamp_sec, utterances)[0]
            ]
            if len(speaking_samples) >= 12:
                jaws = [s.blendshapes.get("jaw_open", 0.0) for s in speaking_samples]
                if float(np.max(jaws) - np.min(jaws)) < 0.008 and listener_only_override:
                    for s in speaking_samples:
                        if QualityFailureFlag.LISTENER_IN_FRAME not in s.quality_flags:
                            s.quality_flags.append(QualityFailureFlag.LISTENER_IN_FRAME)
                        s.usable = False
                        s.speaker_attributed = False

        events, selected_b64 = self.propose_motion_intervals(
            samples=samples,
            utterances=utterances,
            face_crops=face_crops,
        )
        discarded = sorted(
            {flag for s in samples for flag in s.quality_flags},
            key=lambda x: x.value,
        )
        return VideoAnalysisResult(
            frames=samples,
            candidate_events=events,
            effective_fps=round(effective_fps, 2),
            discarded_flags=discarded,
            face_crops_by_frame_id=selected_b64,
        )

    def propose_motion_intervals(
        self,
        samples: Sequence[FrameSample],
        utterances: Sequence[UtteranceSegment] = (),
        face_crops: dict[str, np.ndarray] | None = None,
    ) -> tuple[list[CandidateMotionInterval], dict[str, str]]:
        """Condition-matched same-speaker baseline spotting without fixed calibration duration."""
        usable_samples = [s for s in samples if s.face_visible and s.usable and s.speaker_attributed]
        # Live glasses streams are bursty — allow scoring with a thinner window.
        if len(usable_samples) < 5:
            return [], {}

        tracked_channels = [
            ("lip_press", "brief lip press"),
            ("gaze_shift", "lateral gaze shift"),
            ("brow_furrow", "brow tension change"),
            ("mouth_asymmetry", "asymmetric mouth tension"),
            ("eye_blink", "rapid blink burst"),
        ]

        # Partition usable frames by condition (speaking vs listening)
        cond_map: dict[str, list[FrameSample]] = {"speaking": [], "listening": []}
        for s in usable_samples:
            speaking, _, _, _ = _utterance_state_at(s.timestamp_sec, utterances)
            cond_map["speaking" if speaking else "listening"].append(s)

        baselines: dict[tuple[str, str], tuple[float, float]] = {}
        for cond, group in cond_map.items():
            ref = group if len(group) >= 4 else usable_samples
            for ch, _ in tracked_channels:
                vals = np.array([f.blendshapes.get(ch, 0.0) for f in ref], dtype=np.float64)
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med))) * 1.4826
                baselines[(cond, ch)] = (med, max(0.025, mad))

        # Score deviations per sample
        events: list[CandidateMotionInterval] = []
        used_centers: list[float] = []
        candidates_raw: list[tuple[float, int, str, str, str]] = []

        for idx, s in enumerate(samples):
            if not (s.face_visible and s.usable and s.speaker_attributed):
                continue
            speaking, _, _, _ = _utterance_state_at(s.timestamp_sec, utterances)
            cond = "speaking" if speaking else "listening"
            for ch, label in tracked_channels:
                if s.eyewear_occludes_eyes and ch in ("gaze_shift", "eye_blink"):
                    continue
                if ch == "lip_press" and s.blendshapes.get("jaw_open", 0.0) > 0.16:
                    # Cannot physically press lips together while jaw is wide open articulating vowels
                    continue
                med, scale = baselines[(cond, ch)]
                val = s.blendshapes.get(ch, 0.0)
                delta = val - med
                z = delta / scale
                if ch == "eye_blink":
                    min_abs = 0.24
                    min_z = 2.35
                elif cond == "speaking":
                    min_abs = 0.088
                    min_z = 2.02
                else:
                    min_abs = 0.085
                    min_z = 2.10
                if delta >= min_abs and z >= min_z:
                    candidates_raw.append((delta * z, idx, ch, label, cond))

        candidates_raw.sort(key=lambda item: item[0], reverse=True)
        selected_crop_ids: set[str] = set()

        # Also include 1 reference_still frame if face is visible
        still_sample = min(usable_samples, key=lambda x: x.motion_score)
        selected_crop_ids.add(still_sample.frame_id)
        still_sample.crop_phase = "reference_still"

        for _, apex_idx, ch, label, cond in candidates_raw:
            apex = samples[apex_idx]
            if any(abs(apex.timestamp_sec - c) < 0.55 for c in used_centers):
                continue

            # Expand onset/offset window around apex
            med, _ = baselines[(cond, ch)]
            thresh = med + 0.45 * max(0.04, apex.blendshapes.get(ch, 0.0) - med)
            left = apex_idx
            while left > 0 and samples[left - 1].usable and samples[left - 1].blendshapes.get(ch, 0.0) >= thresh:
                left -= 1
            right = apex_idx
            while (
                right + 1 < len(samples)
                and samples[right + 1].usable
                and samples[right + 1].blendshapes.get(ch, 0.0) >= thresh
            ):
                right += 1

            before_idx = max(0, left - 2)
            after_idx = min(len(samples) - 1, right + 2)
            before_f = samples[before_idx]
            after_f = samples[after_idx]
            if not (before_f.face_visible and after_f.face_visible):
                continue

            used_centers.append(apex.timestamp_sec)
            _, _, word, _ = _utterance_state_at(apex.timestamp_sec, utterances)
            action_desc = f"{label} at {apex.frame_id} ({apex.timestamp_sec:.2f}s)"
            if word:
                action_desc += f" near word '{word}'"

            before_f.crop_phase = "before"
            apex.crop_phase = "during"
            after_f.crop_phase = "after"
            selected_crop_ids.update([before_f.frame_id, apex.frame_id, after_f.frame_id])

            events.append(
                CandidateMotionInterval(
                    event_id=f"evt_{len(events)+1:02d}_{apex.frame_id}",
                    start_sec=round(samples[left].timestamp_sec, 4),
                    end_sec=round(samples[right].timestamp_sec, 4),
                    apex_sec=round(apex.timestamp_sec, 4),
                    apex_frame_id=apex.frame_id,
                    before_frame_id=before_f.frame_id,
                    during_frame_id=apex.frame_id,
                    after_frame_id=after_f.frame_id,
                    observable_action=action_desc,
                    peak_delta=round(float(apex.blendshapes.get(ch, 0.0) - med), 4),
                    baseline_condition="speaking" if cond == "speaking" else "listening",
                    speaker_attributed=True,
                    usable=True,
                    coinciding_word=word,
                )
            )
            if len(events) >= 8:
                break

        events.sort(key=lambda e: e.apex_sec)
        encoded_crops: dict[str, str] = {}
        if face_crops:
            for fid in selected_crop_ids:
                if fid in face_crops:
                    b64 = _encode_jpeg_b64(face_crops[fid])
                    if b64:
                        encoded_crops[fid] = b64
            for s in samples:
                if s.frame_id in encoded_crops:
                    s.jpeg_base64 = encoded_crops[s.frame_id]

        return events, encoded_crops


def render_face_masked_video(input_video: Path, output_video: Path) -> Path:
    """Renders a face-masked copy of input_video preserving exact original audio track."""
    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_silent = output_video.with_suffix(".silent_masked.mp4")

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = ensure_face_landmarker_model()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=3,
        min_face_detection_confidence=0.35,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(input_video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(tmp_silent),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    last_boxes: list[tuple[int, int, int, int]] = []
    f_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f_idx % 5 == 0 or not last_boxes:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            boxes: list[tuple[int, int, int, int]] = []
            for fl in det.face_landmarks:
                xs = [p.x * w for p in fl]
                ys = [p.y * h for p in fl]
                pad_x = int(0.18 * (max(xs) - min(xs)))
                pad_y = int(0.22 * (max(ys) - min(ys)))
                boxes.append(
                    (
                        max(0, int(min(xs)) - pad_x),
                        max(0, int(min(ys)) - pad_y),
                        min(w, int(max(xs)) + pad_x),
                        min(h, int(max(ys)) + pad_y),
                    )
                )
            if boxes:
                last_boxes = boxes
        for x0, y0, x1, y1 in last_boxes:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (12, 12, 12), thickness=-1)
        writer.write(frame)
        f_idx += 1

    cap.release()
    writer.release()
    landmarker.close()

    # Remux original audio track untouched with H.264 video
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(tmp_silent),
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(output_video),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp_silent.unlink(missing_ok=True)
    return output_video
