"""Typed domain models, safety validators, and reveal firewall for Sherlocked."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class ActivityKind(str, Enum):
    GAME = "two_truths_and_a_lie"
    DEAL = "deal"
    PRACTICE = "practice"
    UNCLEAR = "unclear"


class CaptureSourceKind(str, Enum):
    GLASSES = "glasses"
    IPHONE = "iphone"
    REPLAY = "replay"


class AudioRouteKind(str, Enum):
    GLASSES_PRIVATE = "glasses_private"
    EARBUDS_PRIVATE = "earbuds_private"
    SPEAKER_PUBLIC = "speaker_public"
    NONE = "none"

    @property
    def is_private(self) -> bool:
        return self in (AudioRouteKind.GLASSES_PRIVATE, AudioRouteKind.EARBUDS_PRIVATE)


class ModalityMode(str, Enum):
    AUDIO_PLUS_VISIBLE_FACE = "audio_plus_visible_face"
    AUDIO_PLUS_FACE_MASKED_VIDEO = "audio_plus_face_masked_video"
    AUDIO_ONLY = "audio_only"
    TRANSCRIPT_ONLY = "transcript_only"


class QualityFailureFlag(str, Enum):
    LOW_LIGHT = "low_light"
    EYEWEAR_OCCLUSION = "eyewear_occlusion"
    FAST_HEAD_TURN = "fast_head_turn"
    FRAME_LOSS = "frame_loss"
    AUDIO_DRIFT = "audio_drift"
    OVERLAPPING_SPEECH = "overlapping_speech"
    LISTENER_IN_FRAME = "listener_in_frame"


FORBIDDEN_CLINICAL_MORAL_PATTERNS = [
    re.compile(r"\b(pathological|psychopath\w*|sociopath\w*|narcissis\w*)\b", re.IGNORECASE),
    re.compile(r"\b(guilty conscience|immoral|dishonest person|deceitful character)\b", re.IGNORECASE),
    re.compile(r"\b(anxiety disorder|panic disorder|clinical depression|manic|neurotic)\b", re.IGNORECASE),
    re.compile(r"\b(micro-?trauma|criminal intent|malicious|untrustworthy person)\b", re.IGNORECASE),
]


def sanitize_observable_text(text: str, allow_micro_expression_claim: bool = False) -> str:
    cleaned = text.strip()
    for pat in FORBIDDEN_CLINICAL_MORAL_PATTERNS:
        cleaned = pat.sub("observable shift", cleaned)
    if not allow_micro_expression_claim:
        cleaned = re.sub(
            r"\bmicro-?expressions?\b",
            "brief facial movement",
            cleaned,
            flags=re.IGNORECASE,
        )
    return cleaned


class WordTimestamp(BaseModel):
    word: str
    start_sec: float
    end_sec: float
    confidence: float = 0.95
    speaker_id: str = "speaker_primary"


class UtteranceSegment(BaseModel):
    segment_id: str
    speaker_id: str = "speaker_primary"
    text: str
    start_sec: float
    end_sec: float
    overlapping_speech: bool = False
    mean_pitch_hz: float | None = None
    pitch_std_hz: float | None = None
    rms_energy: float | None = None
    pause_before_sec: float = 0.0
    speech_rate_wps: float | None = None
    words: list[WordTimestamp] = Field(default_factory=list)


class FrameSample(BaseModel):
    frame_id: str
    timestamp_sec: float
    crop_phase: Literal["before", "during", "after", "reference_still", "sampled"] = "sampled"
    jpeg_base64: str | None = None
    face_visible: bool = True
    usable: bool = True
    speaker_attributed: bool = True
    faces_detected: int = 1
    luminance: float = 128.0
    head_yaw_deg: float = 0.0
    head_pitch_deg: float = 0.0
    head_turn_deg_per_sec: float = 0.0
    eyewear_detected: bool = False
    eyewear_occludes_eyes: bool = False
    motion_score: float = 0.0
    blendshapes: dict[str, float] = Field(default_factory=dict)
    quality_flags: list[QualityFailureFlag] = Field(default_factory=list)


class CandidateMotionInterval(BaseModel):
    event_id: str
    start_sec: float
    end_sec: float
    apex_sec: float
    apex_frame_id: str
    before_frame_id: str
    during_frame_id: str
    after_frame_id: str
    observable_action: str
    peak_delta: float
    baseline_condition: Literal["speaking", "listening"] = "speaking"
    speaker_attributed: bool = True
    usable: bool = True
    coinciding_word: str | None = None
    independently_annotated_brief_movement: bool = False
    quality_flags: list[QualityFailureFlag] = Field(default_factory=list)

    @field_validator("observable_action")
    @classmethod
    def _clean_action(cls, v: str) -> str:
        return sanitize_observable_text(v, allow_micro_expression_claim=False)


class StorySpan(BaseModel):
    story_number: int = Field(ge=1, le=3)
    speaker_id: str = "speaker_primary"
    status: Literal["complete", "pending"] = "complete"
    start_sec: float
    end_sec: float
    transcript: str
    cited_frame_ids: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    sparse_visual_evidence: bool = False


class GameGuessRecord(BaseModel):
    chosen_story_number: int = Field(ge=1, le=3)
    spoken_whisper: str
    probabilities: dict[str, float]
    confidence: float
    exploratory_disclosure: bool = False
    sparse_visual_disclosure: bool = False
    cited_frame_ids: list[str] = Field(default_factory=list)
    observed_actions: list[str] = Field(default_factory=list)
    comparison_rationale: str
    locked_at_sec: float


class DealCueOutput(BaseModel):
    action: Literal["whisper_question", "silence"]
    identified_term: str | None = None
    spoken_whisper: str | None = None
    confidence: float = 0.0
    cited_frame_ids: list[str] = Field(default_factory=list)
    observable_evidence: str | None = None
    silence_reason: str | None = None


class PracticeFeedbackOutput(BaseModel):
    exercise_complete: bool
    replay_start_sec: float | None = None
    replay_end_sec: float | None = None
    cited_frame_ids: list[str] = Field(default_factory=list)
    observable_moment_summary: str | None = None
    suggested_followup: str | None = None
    confidence: float = 0.0


class SessionAnalyzeRequest(BaseModel):
    session_id: str
    turn_id: str | None = None
    capture_source: CaptureSourceKind = CaptureSourceKind.IPHONE
    audio_route: AudioRouteKind = AudioRouteKind.EARBUDS_PRIVATE
    modality_mode: ModalityMode = ModalityMode.AUDIO_PLUS_VISIBLE_FACE
    is_replay: bool = False
    is_evaluation: bool = False
    request_followup_questions: bool = False
    is_followup_round: bool = False
    effective_fps: float = 24.0
    audio_video_drift_ms: float = 0.0
    utterances: list[UtteranceSegment] = Field(default_factory=list)
    frames: list[FrameSample] = Field(default_factory=list)
    audio_wav_base64: str | None = None
    video_mp4_base64: str | None = None
    extra_metadata: dict[str, Any] = Field(default_factory=dict)

    def assert_reveal_firewall(self) -> None:
        forbidden_keys = {"reveal", "actual_lie", "ground_truth", "lie_index", "lie_story_number", "secret_constraint"}
        leaked = forbidden_keys.intersection({k.lower() for k in self.extra_metadata.keys()})
        if leaked:
            raise ValueError(f"Reveal firewall violation: {sorted(leaked)} must never enter guesser input.")


class ManualCorrectionRequest(BaseModel):
    session_id: str
    corrected_stories: list[StorySpan] | None = None
    revealed_lie_story_number: int | None = Field(default=None, ge=1, le=3)


class SessionAnalyzeResponse(BaseModel):
    result_id: str
    session_id: str
    activity_epoch: int = 1
    activity: ActivityKind
    status_label: str
    execution_mode: Literal["live_gemini", "local_multimodal_engine"]
    models_used: dict[str, str]
    already_delivered: bool = False
    should_speak_privately: bool = False
    present_cue_on_screen: bool = False
    spoken_cue_text: str | None = None
    retry_requested: bool = False
    retry_reason: str | None = None
    stories: list[StorySpan] = Field(default_factory=list)
    initial_guess: GameGuessRecord | None = None
    followup_questions: list[str] = Field(default_factory=list)
    updated_guess: GameGuessRecord | None = None
    revealed_lie_story_number: int | None = None
    deal_output: DealCueOutput | None = None
    practice_output: PracticeFeedbackOutput | None = None
    candidate_events: list[CandidateMotionInterval] = Field(default_factory=list)
    discarded_quality_flags: list[QualityFailureFlag] = Field(default_factory=list)
    effective_fps: float = 24.0
    ModalityUsed: ModalityMode = ModalityMode.AUDIO_PLUS_VISIBLE_FACE
    telemetry: dict[str, Any] = Field(default_factory=dict)
