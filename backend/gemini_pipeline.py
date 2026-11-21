"""Shared 3-stage multimodal pipeline: Gemini 4 Neon -> Gemini 4 Carbon -> Gemini Decisions API."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .audio_analyzer import enrich_utterances_with_audio, transcribe_media_with_whisper
from .config import LearnedWeights, Settings, get_settings
from .schemas import (
    ActivityKind,
    CandidateMotionInterval,
    DealCueOutput,
    FrameSample,
    GameGuessRecord,
    ModalityMode,
    PracticeFeedbackOutput,
    QualityFailureFlag,
    SessionAnalyzeRequest,
    SessionAnalyzeResponse,
    StorySpan,
    UtteranceSegment,
    sanitize_observable_text,
)
from .session_store import SessionRecord, SessionStore
from .vision_tracker import FacialObservationTracker

# Tolerates common front-speaker / glasses-mic ASR misses after soft normalization.
_GAME_RULE_AND_LIE = r"[, ]*(?:(?:and|in|&|an)\s+)?(?:(?:a|one|1|the)\s+)?(?:lie|alive|rely|line|ally)"
_GAME_RULE_BODY = (
    r"(?:let'?s\s+)?play\s+(?:a\s+)?(?:game\s+of\s+)?"
    r"(?:to|too|two|2|true)\s+truths?"
    + _GAME_RULE_AND_LIE
    + r"?"
    + r"|(?:to|too|two|2|true)\s+truths?"
    + _GAME_RULE_AND_LIE
    + r"|two\s+true.{0,24}(?:one|a)\s+lie"
    + r"|three\s+(?:stories|statements).{0,40}one\s+(?:is\s+a\s+)?lie"
    + r"|guess\s+which\s+(?:one|story)\s+is\s+(?:the\s+)?lie"
)
GAME_RULE_PATTERN = re.compile(r"\b(?:" + _GAME_RULE_BODY + r")\b", re.IGNORECASE)


def _find_game_rule(text: str) -> re.Match | None:
    """Pick the invitation match that consumes the most text (avoids stopping at 'truths')."""
    best: re.Match | None = None
    for m in GAME_RULE_PATTERN.finditer(text):
        if best is None or m.end() > best.end():
            best = m
    return best
_ASR_INVITE_FIXES = [
    (re.compile(r"\bto truths\b", re.IGNORECASE), "two truths"),
    (re.compile(r"\btoo truths\b", re.IGNORECASE), "two truths"),
    (re.compile(r"\btwo truth\b", re.IGNORECASE), "two truths"),
    (re.compile(r"\btrue truths\b", re.IGNORECASE), "two truths"),
    (re.compile(r"\btwo truce\b", re.IGNORECASE), "two truths"),
    (re.compile(r"\btwo crews\b", re.IGNORECASE), "two truths"),
    (re.compile(r"\band alive\b", re.IGNORECASE), "and a lie"),
    (re.compile(r"\ban alive\b", re.IGNORECASE), "and a lie"),
    (re.compile(r"\band the lie\b", re.IGNORECASE), "and a lie"),
    (re.compile(r"\bin a lie\b", re.IGNORECASE), "and a lie"),
    (re.compile(r"\band rely\b", re.IGNORECASE), "and a lie"),
    (re.compile(r"\band a line\b", re.IGNORECASE), "and a lie"),
    (re.compile(r"\band ally\b", re.IGNORECASE), "and a lie"),
    (re.compile(r"\btwo truths and lie\b", re.IGNORECASE), "two truths and a lie"),
    (re.compile(r"\bto truths and a lie\b", re.IGNORECASE), "two truths and a lie"),
]
PRACTICE_RULE_PATTERN = re.compile(
    r"\b(role-?play|rehears(?:e|al|ing)|practice (?:pitch|negotiation|session|round|run)|coach me|mock pitch|let'?s practice)\b",
    re.IGNORECASE,
)
PRACTICE_END_PATTERN = re.compile(
    r"\b(end (?:role-?play|scene|practice)|stop role-?play|how (?:was|did) that(?: sound)?|pause coaching|wrap up the rehearsal)\b",
    re.IGNORECASE,
)
DEAL_TERM_PATTERN = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|next week|q[1-4]|deadline|deliver(?:y)?|ship(?:ping)?|timeline|"
    r"\$\s?\d[\d,]*|\d+\s?(?:k|percent|%|seats|months|days|weeks)|budget|pricing|discount|exclusiv\w*|sla|contract|commit\w*|milestone|sign-?off)\b",
    re.IGNORECASE,
)
FOLLOWUP_REQUEST_PATTERN = re.compile(
    r"\b(follow-?up questions?|ask (?:me |you |him |her )?questions?|before (?:the |i )?reveal|interrogate)\b",
    re.IGNORECASE,
)
REVEAL_SPEECH_PATTERN = re.compile(
    r"\b(?:the lie (?:was|is)|lie is|actually fake was|made up)\s*(?:story\s*|number\s*|#)?(one|two|too|three|1|2|3)\b"
    r"|\b(?:number|story)\s*(one|two|too|three|1|2|3)\s*(?:was|is)\s*(?:the\s*)?(?:a\s*)?lie\b",
    re.IGNORECASE,
)
ORDINAL_SPLITS = [
    (1, re.compile(r"^(?:(?:story|number|statement)\s+)?(?:one|1|first(?:ly)?)\b[,:.\s-]*", re.IGNORECASE)),
    (2, re.compile(r"^(?:(?:story|number|statement)\s+)?(?:two|too|2|second(?:ly)?)\b[,:.\s-]*", re.IGNORECASE)),
    (3, re.compile(r"^(?:(?:story|number|statement)\s+)?(?:three|3|third(?:ly)?)\b[,:.\s-]*", re.IGNORECASE)),
]
INLINE_ORDINAL_SPLITS = [
    (1, re.compile(
        r"(?:(?:^|[.!?;:]\s+)(?:(?:story|number|statement)\s+)?|\b(?:story|number|statement)\s+)"
        r"(?:one|1|first(?:ly)?)\b[,:.\s-]+",
        re.IGNORECASE,
    )),
    (2, re.compile(
        r"(?:(?:^|[.!?;:]\s+)(?:(?:story|number|statement)\s+)?|\b(?:story|number|statement)\s+)"
        r"(?:two|too|2|second(?:ly)?)\b[,:.\s-]+"
        r"|\b(?:two|too|2)\s+(?=I\b|I'm\b|I\s|my\b)",
        re.IGNORECASE,
    )),
    (3, re.compile(
        r"(?:(?:^|[.!?;:]\s+)(?:(?:story|number|statement)\s+)?|\b(?:story|number|statement)\s+)"
        r"(?:three|3|third(?:ly)?)\b[,:.\s-]+"
        r"|\b(?:three|3)\s+(?=I\b|I'm\b|I\s|my\b)",
        re.IGNORECASE,
    )),
]
# Looser markers for a single mega-transcript with no clean sentence breaks.
_LOOSE_STORY_MARKERS = [
    (1, re.compile(r"(?:^|\s)(?:(?:story|number|statement)\s+)?(?:one|1|first(?:ly)?)\b(?:\s*[,:\-–—]|\s+)", re.IGNORECASE)),
    (2, re.compile(r"(?:^|\s)(?:(?:story|number|statement)\s+)?(?:two|too|2|second(?:ly)?)\b(?:\s*[,:\-–—]|\s+)", re.IGNORECASE)),
    (3, re.compile(r"(?:^|\s)(?:(?:story|number|statement)\s+)?(?:three|3|third(?:ly)?)\b(?:\s*[,:\-–—]|\s+)", re.IGNORECASE)),
]
HEDGING_WORDS = re.compile(
    r"\b(kind of|sort of|basically|probably|around|may\s?be|i guess|at some point|somewhere|stuff like that|or something|pretty much)\b",
    re.IGNORECASE,
)
CONCRETE_ANCHOR_WORDS = re.compile(
    r"\b(19\d\d|20\d\d|\d+\s*(?:hours?|miles?|minutes?|years?|days?|feet|degrees|dollars?)|january|february|march|april|may|june|july|august|september|october|november|december|hospital|airport|marathon|university|kitchen|highway|passport|stitches|trophy)\b",
    re.IGNORECASE,
)


def _word_to_story_num(token: str) -> int | None:
    mapping = {"1": 1, "one": 1, "2": 2, "two": 2, "too": 2, "3": 3, "three": 3}
    return mapping.get(token.lower().strip())


def _normalize_front_speaker_asr(text: str) -> str:
    """Repair glasses-mic mishears of the game invitation before matching/scoring."""
    out = text
    for pattern, repl in _ASR_INVITE_FIXES:
        out = pattern.sub(repl, out)
    return out


def _normalize_utterances_asr(utterances: Sequence[UtteranceSegment]) -> list[UtteranceSegment]:
    fixed: list[UtteranceSegment] = []
    for u in utterances:
        new_text = _normalize_front_speaker_asr(u.text)
        if new_text != u.text:
            fixed.append(u.model_copy(update={"text": new_text}))
        else:
            fixed.append(u)
    return fixed


def _slice_utt(u: UtteranceSegment, text: str, start_frac: float, end_frac: float) -> UtteranceSegment:
    dur = max(0.1, u.end_sec - u.start_sec)
    return u.model_copy(
        update={
            "text": text.strip().rstrip(".") + ("" if text.strip().endswith((".", "!", "?")) else "."),
            "start_sec": round(u.start_sec + start_frac * dur, 3),
            "end_sec": round(u.start_sec + end_frac * dur, 3),
        }
    )


def _carve_three_stories_from_utterances(
    player_utts: Sequence[UtteranceSegment],
    existing: dict[int, list[UtteranceSegment]] | None = None,
) -> dict[int, list[UtteranceSegment]]:
    """
    Recover three stories when live ASR dumps the whole round into one/two utterances.
    Priority: loose ordinals → sentence groups → pause-separated utterances → word thirds.
    """
    if not player_utts:
        return {}

    # 2 pause-separated utterances already? Keep story1, split the longer remainder.
    if len(player_utts) == 2 and (not existing or len(existing) <= 1):
        first, second = player_utts[0], player_utts[1]
        gap = second.start_sec - first.end_sec
        if gap >= 0.35 and len(second.text.split()) >= 8:
            sub = _carve_three_stories_from_utterances([second], existing=None)
            if len(sub) >= 2:
                out = {1: [first]}
                # Remap sub 1/2 → 2/3 (or keep numbers if already 2/3)
                nums = sorted(sub.keys())
                if nums == [1, 2]:
                    out[2] = sub[1]
                    out[3] = sub[2]
                else:
                    out.update(sub)
                return out

    joined = " ".join(u.text.strip() for u in player_utts).strip()
    if len(joined.split()) < 9:
        return dict(existing or {})

    t0 = player_utts[0].start_sec
    t1 = player_utts[-1].end_sec
    dur = max(0.3, t1 - t0)

    def _pack(parts: list[str]) -> dict[int, list[UtteranceSegment]]:
        clean = [p.strip() for p in parts if p and len(p.strip().split()) >= 3]
        if len(clean) < 3:
            return {}
        # Keep the three most substantial chunks in order (first / middle / last).
        if len(clean) > 3:
            clean = [clean[0], " ".join(clean[1:-1]), clean[-1]]
        out: dict[int, list[UtteranceSegment]] = {}
        cursor = 0
        total = max(1, len(joined))
        for idx, text in enumerate(clean[:3], start=1):
            # Approximate time by character offset in the joined string.
            pos = joined.lower().find(text[: min(24, len(text))].lower(), cursor)
            if pos < 0:
                pos = cursor
            end_pos = pos + len(text)
            cursor = end_pos
            start_frac = pos / total
            end_frac = min(1.0, end_pos / total)
            synthetic = player_utts[0].model_copy(
                update={
                    "text": text if text.endswith((".", "!", "?")) else text + ".",
                    "start_sec": round(t0 + start_frac * dur, 3),
                    "end_sec": round(t0 + end_frac * dur, 3),
                    "pause_before_sec": 0.2 if idx > 1 else player_utts[0].pause_before_sec,
                }
            )
            out[idx] = [synthetic]
        return out if len(out) == 3 else {}

    # 1) Loose ordinal markers across the full transcript.
    hits: list[tuple[int, int]] = []
    for num, pat in _LOOSE_STORY_MARKERS:
        m = pat.search(joined)
        if m:
            hits.append((m.start(), num))
    hits = sorted({(pos, num) for pos, num in hits}, key=lambda x: x[0])
    # Prefer increasing 1→2→3 order; drop out-of-order duplicates.
    ordered: list[tuple[int, int]] = []
    seen: set[int] = set()
    for pos, num in hits:
        if num in seen:
            continue
        if ordered and num < ordered[-1][1]:
            continue
        ordered.append((pos, num))
        seen.add(num)
    if len(ordered) >= 2:
        parts: list[str] = []
        # Leading text before first marker → story 1 if marker isn't 1 at start.
        if ordered[0][0] > 10 and ordered[0][1] != 1:
            lead = joined[: ordered[0][0]].strip()
            if len(lead.split()) >= 3:
                parts.append(lead)
        for i, (pos, _) in enumerate(ordered):
            end = ordered[i + 1][0] if i + 1 < len(ordered) else len(joined)
            chunk = joined[pos:end].strip()
            if chunk:
                parts.append(chunk)
        packed = _pack(parts)
        if len(packed) == 3:
            return packed

    # 2) Sentence split into ~3 groups.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", joined) if s.strip()]
    if len(sentences) >= 3:
        # Greedy pack into 3 buckets with balanced word counts.
        total_words = sum(len(s.split()) for s in sentences) or 1
        target = total_words / 3.0
        groups: list[list[str]] = [[], [], []]
        gi = 0
        count = 0
        for s in sentences:
            groups[gi].append(s)
            count += len(s.split())
            if gi < 2 and count >= target * 0.85:
                gi += 1
                count = 0
        packed = _pack([" ".join(g) for g in groups])
        if len(packed) == 3:
            return packed

    # 2b) First-person / discourse clause splits — live ASR often has no punctuation.
    # Require ≥5 words per claim and ≥18 words total so a short mid-story "I… I…"
    # does not invent a finished round, but three real claims still lock a guess.
    clause_pat = re.compile(
        r"(?:^|\s+)(?="
        r"I\s+(?:(?:never|once|used|just|still|always|also|even)\s+)?[a-z]"
        r"|I'?m\s+"
        r"|(?:and\s+)?(?:then|finally|lastly|also|another(?:\s+one)?)\b"
        r"|(?:first(?:ly)?|second(?:ly)?|third(?:ly)?)\b"
        r")",
        re.IGNORECASE,
    )
    starts = [m.start() for m in clause_pat.finditer(" " + joined)]
    starts = [max(0, s - 1) for s in starts]
    starts = sorted({s for s in starts if s < len(joined)})
    if len(starts) >= 3:
        parts = []
        for i, pos in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(joined)
            chunk = joined[pos:end].strip(" ,")
            if chunk and len(chunk.split()) >= 5:
                parts.append(chunk)
        if len(parts) > 3:
            short = [p for p in parts if 5 <= len(p.split()) <= 22]
            parts = short[:3] if len(short) >= 3 else [parts[0], " ".join(parts[1:-1]), parts[-1]]
        if len(parts) >= 3 and sum(len(p.split()) for p in parts[:3]) >= 18:
            packed = _pack(parts[:3])
            if len(packed) == 3:
                return packed

    # 3) Three+ pause-separated utterances without ordinals — map by order.
    if len(player_utts) >= 3:
        substantive = [u for u in player_utts if len(u.text.split()) >= 4]
        if len(substantive) >= 3:
            return {
                1: [substantive[0]],
                2: [substantive[1]],
                3: [substantive[2]],
            }

    # No equal word-thirds — that mis-bucketed the hedged lie into Story #1 on device.
    return dict(existing or {})


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps) or 1.0
    probs = [e / total for e in exps]
    # Normalize rounding to sum to 1.0000
    rounded = [round(p, 4) for p in probs]
    diff = round(1.0 - sum(rounded), 4)
    top_i = int(max(range(len(rounded)), key=lambda i: rounded[i]))
    rounded[top_i] = round(rounded[top_i] + diff, 4)
    return rounded


def _strip_reveal_utterances(
    utterances: Sequence[UtteranceSegment],
) -> tuple[list[UtteranceSegment], int | None, list[UtteranceSegment]]:
    """Splits utterances at any spoken reveal so reveal NEVER enters guesser inputs."""
    pre_reveal: list[UtteranceSegment] = []
    post_reveal: list[UtteranceSegment] = []
    revealed_num: int | None = None

    for u in utterances:
        if revealed_num is not None:
            post_reveal.append(u)
            continue
        m = REVEAL_SPEECH_PATTERN.search(u.text)
        if m:
            tok = m.group(1) or m.group(2)
            revealed_num = _word_to_story_num(tok) if tok else None
            # Keep any story sentence spoken BEFORE the reveal clause in the same utterance, stripping the reveal clause itself
            prefix = u.text[: m.start()].strip()
            prefix = re.sub(r"\b(?:time to|okay|alright|now)\s+(?:lock|make|take)?[^.!?]*$", "", prefix, flags=re.IGNORECASE).strip()
            if len(prefix.split()) >= 5:
                frac = max(0.2, len(prefix) / float(max(1, len(u.text))))
                trimmed_end = u.start_sec + frac * max(0.1, u.end_sec - u.start_sec)
                pre_reveal.append(u.model_copy(update={"text": prefix, "end_sec": round(trimmed_end, 3)}))
            continue
        pre_reveal.append(u)

    return pre_reveal, revealed_num, post_reveal


class SherlockedPipeline:
    """Unified pipeline driving research benchmarks, replay fixtures, iPhone fallback, and Ray-Ban glasses."""

    def __init__(
        self,
        settings: Settings | None = None,
        store: SessionStore | None = None,
        weights: LearnedWeights | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or SessionStore(self.settings.db_path)
        self.weights = weights or LearnedWeights.load()
        self.tracker = FacialObservationTracker()

    def reload_weights(self) -> None:
        self.weights = LearnedWeights.load()

    def _compute_result_id(
        self,
        req: SessionAnalyzeRequest,
        epoch: int,
        activity: ActivityKind,
        utterances: Sequence[UtteranceSegment],
    ) -> str:
        canonical = {
            "session_id": req.session_id,
            "turn_id": req.turn_id or "",
            "epoch": epoch,
            "activity": activity.value,
            "modality": req.modality_mode.value,
            "followup_req": req.request_followup_questions,
            "followup_round": req.is_followup_round,
            "utt_hash": hashlib.sha256(
                "|".join(f"{u.start_sec:.2f}:{u.end_sec:.2f}:{u.text.strip()}" for u in utterances).encode("utf-8")
            ).hexdigest()[:16],
        }
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        return f"res_{req.session_id}_{epoch}_{digest}"

    def _prepare_media_inputs(
        self, req: SessionAnalyzeRequest
    ) -> tuple[list[UtteranceSegment], list[FrameSample], list[CandidateMotionInterval], list[QualityFailureFlag], float]:
        utterances = list(req.utterances)
        frames = list(req.frames)
        effective_fps = req.effective_fps
        discarded_flags: set[QualityFailureFlag] = set()

        tmp_files: list[Path] = []
        try:
            wav_path: Path | None = None
            if req.audio_wav_base64:
                fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                fd.write(base64.b64decode(req.audio_wav_base64))
                fd.close()
                wav_path = Path(fd.name)
                tmp_files.append(wav_path)

            video_path: Path | None = None
            if req.video_mp4_base64:
                fd = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                fd.write(base64.b64decode(req.video_mp4_base64))
                fd.close()
                video_path = Path(fd.name)
                tmp_files.append(video_path)

            if not utterances and (wav_path or video_path):
                media_for_asr = wav_path or video_path
                assert media_for_asr is not None
                utterances = transcribe_media_with_whisper(media_for_asr)
            elif utterances and wav_path and req.modality_mode != ModalityMode.TRANSCRIPT_ONLY:
                utterances = enrich_utterances_with_audio(utterances, wav_path)

            # Enforce modality ablations cleanly
            if req.modality_mode == ModalityMode.TRANSCRIPT_ONLY:
                utterances = [
                    u.model_copy(
                        update={
                            "mean_pitch_hz": None,
                            "pitch_std_hz": None,
                            "rms_energy": None,
                            "pause_before_sec": 0.0,
                            "speech_rate_wps": None,
                        }
                    )
                    for u in utterances
                ]
                frames = []
                return utterances, [], [], [], effective_fps

            if req.modality_mode in (
                ModalityMode.AUDIO_ONLY,
                ModalityMode.AUDIO_PLUS_FACE_MASKED_VIDEO,
            ):
                # Omit facial observations in audio_only and face_masked runs per spec line 51
                return utterances, [], [], [], effective_fps

            # AUDIO_PLUS_VISIBLE_FACE
            if not frames and video_path is not None:
                vres = self.tracker.analyze_video_file(
                    video_path=video_path,
                    utterances=utterances,
                    audio_video_drift_ms=req.audio_video_drift_ms,
                )
                frames = vres.frames
                effective_fps = vres.effective_fps
                discarded_flags.update(vres.discarded_flags)
                return utterances, frames, vres.candidate_events, sorted(discarded_flags, key=lambda x: x.value), effective_fps

            # Pre-extracted frames sent from iOS Vision tracker (glasses/iPhone live path).
            # Soft wearable flags are telemetry-only — do not starve lie/deal scoring.
            soft_live = {
                QualityFailureFlag.FRAME_LOSS,
                QualityFailureFlag.FAST_HEAD_TURN,
                QualityFailureFlag.EYEWEAR_OCCLUSION,
                QualityFailureFlag.LISTENER_IN_FRAME,
                QualityFailureFlag.LOW_LIGHT,
            }
            hard_live = {
                QualityFailureFlag.AUDIO_DRIFT,
                QualityFailureFlag.OVERLAPPING_SPEECH,
            }
            for f in frames:
                if abs(req.audio_video_drift_ms) > 140.0 and QualityFailureFlag.AUDIO_DRIFT not in f.quality_flags:
                    f.quality_flags.append(QualityFailureFlag.AUDIO_DRIFT)
                flags = set(f.quality_flags)
                hard = flags.intersection(hard_live)
                # Extreme low-light / no-face stay unusable; soft wearable flags do not.
                extreme_dark = QualityFailureFlag.LOW_LIGHT in flags and float(f.luminance or 128.0) < 28.0
                if hard or extreme_dark or not f.face_visible:
                    f.usable = False
                    f.speaker_attributed = False
                elif f.face_visible and (not flags or flags.issubset(soft_live) or f.usable):
                    f.usable = True
                    f.speaker_attributed = True
                discarded_flags.update(f.quality_flags)

            events, _ = self.tracker.propose_motion_intervals(frames, utterances)
            return utterances, frames, events, sorted(discarded_flags, key=lambda x: x.value), effective_fps
        finally:
            for p in tmp_files:
                p.unlink(missing_ok=True)

    def _detect_activity_and_epoch(
        self,
        session: SessionRecord,
        pre_reveal_utts: Sequence[UtteranceSegment],
        revealed_num: int | None,
        post_reveal_utts: Sequence[UtteranceSegment],
    ) -> tuple[ActivityKind, int, Sequence[UtteranceSegment]]:
        """Determines active context without a mode picker, handling post-reveal transitions."""
        if revealed_num is not None and session.initial_guess is not None:
            session.game_revealed = True
            session.revealed_lie_story_number = revealed_num

        active_utts = post_reveal_utts if (session.game_revealed and post_reveal_utts) else pre_reveal_utts
        joined = " ".join(u.text for u in active_utts).strip()

        # If transitioning after a revealed game into a new activity in the same session:
        if session.game_revealed and post_reveal_utts:
            if PRACTICE_RULE_PATTERN.search(joined):
                if session.current_activity != ActivityKind.PRACTICE:
                    session.activity_epoch += 1
                    session.current_activity = ActivityKind.PRACTICE
                return ActivityKind.PRACTICE, session.activity_epoch, post_reveal_utts
            if DEAL_TERM_PATTERN.search(joined):
                if session.current_activity != ActivityKind.DEAL:
                    session.activity_epoch += 1
                    session.current_activity = ActivityKind.DEAL
                return ActivityKind.DEAL, session.activity_epoch, post_reveal_utts

        # Practice ALWAYS takes precedence over deal language inside an exercise!
        full_pre_text = " ".join(u.text for u in pre_reveal_utts).strip()
        if PRACTICE_RULE_PATTERN.search(full_pre_text):
            session.current_activity = ActivityKind.PRACTICE
            return ActivityKind.PRACTICE, session.activity_epoch, pre_reveal_utts

        # Keep a recognized game stable through the reveal
        if session.current_activity == ActivityKind.GAME and not (session.game_revealed and post_reveal_utts):
            return ActivityKind.GAME, session.activity_epoch, pre_reveal_utts

        if GAME_RULE_PATTERN.search(full_pre_text):
            session.current_activity = ActivityKind.GAME
            return ActivityKind.GAME, session.activity_epoch, pre_reveal_utts

        if DEAL_TERM_PATTERN.search(full_pre_text):
            session.current_activity = ActivityKind.DEAL
            return ActivityKind.DEAL, session.activity_epoch, pre_reveal_utts

        session.current_activity = ActivityKind.UNCLEAR
        return ActivityKind.UNCLEAR, session.activity_epoch, pre_reveal_utts

    def _segment_three_stories(
        self,
        utterances: Sequence[UtteranceSegment],
        events: Sequence[CandidateMotionInterval],
        frames: Sequence[FrameSample],
    ) -> list[StorySpan]:
        if not utterances:
            return []

        # Match the invitation across pauses, including a first story in the same segment.
        joined = " ".join(u.text for u in utterances)
        rule = _find_game_rule(joined)
        if rule is None:
            return []
        story_utts: list[UtteranceSegment] = []
        offset = 0
        for u in utterances:
            start = max(0, rule.end() - offset)
            offset += len(u.text) + 1
            tail = u.text[start:].strip(" .,:;-")
            # Invite-only utterance (e.g. "let's play two truths and a lie") — skip.
            if not tail:
                continue
            if FOLLOWUP_REQUEST_PATTERN.search(tail):
                break
            if start:
                if len(tail.split()) < 3 and not any(p.match(tail) for _, p in ORDINAL_SPLITS):
                    continue
                fraction = start / max(1, len(u.text))
                u = u.model_copy(update={"text": tail, "start_sec": u.start_sec + fraction * (u.end_sec - u.start_sec)})
            # Drop residual invite crumbs that leaked past a short regex match.
            if GAME_RULE_PATTERN.fullmatch(tail.strip(" .")) or re.fullmatch(
                r"(?:and\s+)?(?:a\s+)?lie\.?", tail.strip(), flags=re.IGNORECASE
            ):
                continue
            story_utts.append(u)

        if not story_utts:
            return []

        # Dominant player speaker_id
        speaker_counts: dict[str, int] = {}
        for u in story_utts:
            speaker_counts[u.speaker_id] = speaker_counts.get(u.speaker_id, 0) + len(u.text.split())
        player_id = max(speaker_counts, key=speaker_counts.get)
        player_utts = [u for u in story_utts if u.speaker_id == player_id]

        # Try explicit ordinal markers first across player utterances (including inline multi-story segments)
        buckets: dict[int, list[UtteranceSegment]] = {}
        current_num = 0
        for u in player_utts:
            stripped = u.text.strip()
            # Check if a single Whisper utterance spans multiple sentence-initial ordinals (e.g. "...rainstorm. Two, in November...")
            inline_hits: list[tuple[int, int]] = []
            for num, ipat in INLINE_ORDINAL_SPLITS:
                m = ipat.search(stripped)
                if m:
                    inline_hits.append((m.start(), num))
            inline_hits.sort(key=lambda x: x[0])

            # One utterance often carries story 1+2+3 with ordinals mid-stream.
            if len(inline_hits) >= 2 or (
                len(inline_hits) == 1 and len({n for _, n in inline_hits}) == 1 and inline_hits[0][0] > 0
                and current_num in (0, 1, 2)
            ):
                dur = max(0.1, u.end_sec - u.start_sec)
                tot_len = max(1, len(stripped))
                # If the first hit isn't at the start, keep the leading clause in the current story.
                work_hits = list(inline_hits)
                if work_hits and work_hits[0][0] > 8 and current_num in (0, 1, 2, 3):
                    lead_num = current_num if current_num in (1, 2, 3) else 1
                    lead_txt = stripped[: work_hits[0][0]].strip(" .")
                    if lead_txt and len(lead_txt.split()) >= 3:
                        buckets.setdefault(lead_num, []).append(
                            u.model_copy(
                                update={
                                    "text": lead_txt + ".",
                                    "end_sec": u.start_sec + (work_hits[0][0] / tot_len) * dur,
                                }
                            )
                        )
                        current_num = lead_num
                for idx_h, (pos, num) in enumerate(work_hits):
                    end_pos = work_hits[idx_h + 1][0] if (idx_h + 1) < len(work_hits) else tot_len
                    sub_txt = stripped[pos:end_pos].strip(" .")
                    if sub_txt:
                        s_t = u.start_sec + (pos / tot_len) * dur
                        e_t = u.start_sec + (end_pos / tot_len) * dur
                        buckets.setdefault(num, []).append(
                            u.model_copy(update={"text": sub_txt + ".", "start_sec": s_t, "end_sec": e_t})
                        )
                        current_num = num
                continue

            matched_ord = None
            for num, pat in ORDINAL_SPLITS:
                if pat.search(stripped):
                    matched_ord = num
                    break
            if matched_ord is None and len(inline_hits) == 1 and inline_hits[0][0] > 8:
                # Utterance starts by finishing previous story, then introduces next ordinal after punctuation
                pos, next_num = inline_hits[0]
                dur = max(0.1, u.end_sec - u.start_sec)
                tot_len = max(1, len(stripped))
                prev_txt = stripped[:pos].strip()
                next_txt = stripped[pos:].strip()
                mid_t = u.start_sec + (pos / tot_len) * dur
                if prev_txt and current_num in (1, 2, 3):
                    buckets.setdefault(current_num, []).append(
                        u.model_copy(update={"text": prev_txt, "end_sec": mid_t})
                    )
                current_num = next_num
                if next_txt and current_num in (1, 2, 3):
                    buckets.setdefault(current_num, []).append(
                        u.model_copy(update={"text": next_txt, "start_sec": mid_t})
                    )
                continue

            if matched_ord is not None:
                current_num = matched_ord
            elif current_num == 0:
                current_num = 1
            if current_num in (1, 2, 3):
                buckets.setdefault(current_num, []).append(u)

        # Fallback A: speaker or ASR omitted "three" between story 2 and story 3, leaving a clear prosodic pause
        if set(buckets.keys()) == {1, 2}:
            substantive_2 = [
                (i, seg) for i, seg in enumerate(buckets[2]) if len(seg.text.split()) >= 5
            ]
            if len(substantive_2) >= 2:
                prev_i, prev_seg = substantive_2[-2]
                last_i, last_seg = substantive_2[-1]
                if (last_seg.start_sec - prev_seg.end_sec) >= 0.65:
                    buckets[3] = buckets[2][last_i:]
                    buckets[2] = buckets[2][:last_i]

        # Fallback B: speaker told 3 separate utterances without saying "one/two/three"
        if len(buckets) < 3 and len(player_utts) >= 3:
            # Take the last 3 substantive utterances (skip invite crumbs already filtered).
            substantive = [u for u in player_utts if len(u.text.split()) >= 3][-3:]
            if len(substantive) >= 3:
                buckets = {
                    1: [substantive[0]],
                    2: [substantive[1]],
                    3: [substantive[2]],
                }
            elif len(buckets) <= 1 and len(player_utts) >= 3:
                buckets = {
                    1: [player_utts[0]],
                    2: [player_utts[1]],
                    3: [player_utts[2]],
                }

        # Fallback C: only when ordinal structure is missing (≤1 bucket).
        # Never invent story 3 from a clear {1,2}-only round — wait for more speech.
        if len(buckets) <= 1:
            carved = _carve_three_stories_from_utterances(player_utts, existing=buckets)
            if len(carved) >= 3:
                buckets = carved
            elif len(carved) > len(buckets):
                buckets = carved

        usable_frame_ids = {f.frame_id for f in frames if f.face_visible and f.usable and f.speaker_attributed}
        stories: list[StorySpan] = []
        for num in sorted(buckets.keys()):
            if num not in (1, 2, 3):
                continue
            seg_list = buckets[num]
            text = " ".join(s.text.strip() for s in seg_list).strip()
            start_s = seg_list[0].start_sec
            end_s = seg_list[-1].end_sec
            has_overlap = any(s.overlapping_speech for s in seg_list)
            word_cnt = len(text.split())
            # Require a real clause — tiny word-third crumbs must not lock a guess.
            complete = (word_cnt >= 6) and not has_overlap and not text.endswith("...")

            story_events = [
                e
                for e in events
                if e.usable
                and e.speaker_attributed
                and (start_s - 0.30) <= e.apex_sec <= (end_s + 0.40)
                and e.apex_frame_id in usable_frame_ids
            ]
            cited = [e.apex_frame_id for e in story_events]
            obs = [
                sanitize_observable_text(e.observable_action, self.weights.allow_micro_expression_claim)
                for e in story_events
            ]
            stories.append(
                StorySpan(
                    story_number=num,
                    speaker_id=player_id,
                    status="complete" if complete else "pending",
                    start_sec=round(start_s, 3),
                    end_sec=round(end_s, 3),
                    transcript=text,
                    cited_frame_ids=cited,
                    observations=obs,
                    sparse_visual_evidence=(len(story_events) == 0),
                )
            )
        return stories

    def _score_stories_multimodal(
        self,
        stories: list[StorySpan],
        utterances: Sequence[UtteranceSegment],
        events: Sequence[CandidateMotionInterval],
        frames: Sequence[FrameSample],
        modality_mode: ModalityMode,
        is_followup: bool = False,
    ) -> GameGuessRecord:
        """Within-speaker comparison across lexical specificity, acoustic prosody, and usable facial observations."""
        assert len(stories) == 3

        lex_raw: list[float] = []
        voc_raw: list[float] = []
        fac_raw: list[float] = []
        per_story_notes: list[str] = []

        # Speaker baseline prosody across the 3 stories
        story_utt_groups: list[list[UtteranceSegment]] = []
        for st in stories:
            matched = [
                u
                for u in utterances
                if u.end_sec >= (st.start_sec - 0.1) and u.start_sec <= (st.end_sec + 0.1)
            ]
            story_utt_groups.append(matched)

        all_pitches = [
            u.mean_pitch_hz
            for grp in story_utt_groups
            for u in grp
            if u.mean_pitch_hz and u.mean_pitch_hz > 50.0
        ]
        mean_speaker_pitch = sum(all_pitches) / len(all_pitches) if all_pitches else 0.0
        all_wps = [
            u.speech_rate_wps
            for grp in story_utt_groups
            for u in grp
            if u.speech_rate_wps and u.speech_rate_wps > 0.2
        ]
        mean_speaker_wps = sum(all_wps) / len(all_wps) if all_wps else 2.5

        usable_visible_frames = [f for f in frames if f.face_visible and f.usable and f.speaker_attributed]

        for idx, st in enumerate(stories):
            txt = st.transcript
            hedges = len(HEDGING_WORDS.findall(txt))
            anchors = len(CONCRETE_ANCHOR_WORDS.findall(txt))
            words = txt.split()
            # Higher score => stronger fabrication candidate relative to sibling stories.
            # Hedge weight is intentionally strong — live face blips on story 1 must not
            # outweigh clear "kind of / probably / somehow" language on the real lie.
            l_score = (
                0.95 * hedges
                - 0.42 * anchors
                + (0.22 if len(words) < 9 else 0.0)
                + (0.18 if hedges >= 2 else 0.0)
            )
            lex_raw.append(l_score)

            # Prosody score
            if modality_mode == ModalityMode.TRANSCRIPT_ONLY:
                v_score = 0.0
            else:
                grp = story_utt_groups[idx]
                max_pause = max((u.pause_before_sec for u in grp), default=0.0)
                pitches = [u.mean_pitch_hz for u in grp if u.mean_pitch_hz and u.mean_pitch_hz > 50.0]
                pitch_shift = (
                    abs((sum(pitches) / len(pitches)) - mean_speaker_pitch) / max(80.0, mean_speaker_pitch)
                    if pitches and mean_speaker_pitch > 0
                    else 0.0
                )
                wps_vals = [u.speech_rate_wps for u in grp if u.speech_rate_wps]
                wps_shift = (
                    abs((sum(wps_vals) / len(wps_vals)) - mean_speaker_wps) / max(1.0, mean_speaker_wps)
                    if wps_vals
                    else 0.0
                )
                v_score = 0.85 * min(2.0, max_pause) + 1.6 * pitch_shift + 0.9 * wps_shift
            voc_raw.append(v_score)

            # Facial observation score
            if modality_mode != ModalityMode.AUDIO_PLUS_VISIBLE_FACE:
                f_score = 0.0
            else:
                st_events = [
                    e
                    for e in events
                    if e.usable
                    and e.speaker_attributed
                    and (st.start_sec - 0.30) <= e.apex_sec <= (st.end_sec + 0.40)
                ]
                f_score = sum(e.peak_delta * 3.2 for e in st_events)
            fac_raw.append(f_score)

            story_grp = story_utt_groups[idx]
            note_parts = [f"Story {st.story_number}:"]
            if hedges:
                note_parts.append(f"{hedges} qualifier phrase(s)")
            if anchors:
                note_parts.append(f"{anchors} concrete detail anchor(s)")
            if modality_mode != ModalityMode.TRANSCRIPT_ONLY and story_grp:
                note_parts.append(f"onset pause {max((u.pause_before_sec for u in story_grp), default=0.0):.2f}s")
            if st.observations:
                note_parts.append("; ".join(st.observations))
            elif modality_mode == ModalityMode.AUDIO_PLUS_VISIBLE_FACE and usable_visible_frames:
                note_parts.append("visible low-motion / straight face")
            per_story_notes.append(" ".join(note_parts))

        # Incorporate post-initial-guess follow-up answers if this is an updated guess round
        if is_followup:
            last_story_end = stories[-1].end_sec
            followup_utts = [u for u in utterances if u.start_sec > last_story_end + 0.1]
            for fu in followup_utts:
                if FOLLOWUP_REQUEST_PATTERN.search(fu.text):
                    continue
                for idx in range(3):
                    num = idx + 1
                    if re.search(rf"\b(?:story|number|#)?\s*(?:{num}|{'one' if num==1 else 'two' if num==2 else 'three'})\b", fu.text, re.IGNORECASE):
                        fu_hedges = len(HEDGING_WORDS.findall(fu.text))
                        fu_anchors = len(CONCRETE_ANCHOR_WORDS.findall(fu.text))
                        lex_raw[idx] += 0.65 * fu_hedges - 0.50 * fu_anchors + 0.55 * fu.pause_before_sec

        w_lex = self.weights.lexical_consistency_weight
        w_voc = 0.0 if modality_mode == ModalityMode.TRANSCRIPT_ONLY else self.weights.vocal_prosody_weight
        # Face only ranks when ≥2 stories show usable motion — a single early glance
        # at the phone otherwise falsely crowns Story #1 as the lie.
        face_hits = sum(1 for f in fac_raw if f > 0.08)
        w_fac = (
            self.weights.facial_evidence_weight
            if modality_mode == ModalityMode.AUDIO_PLUS_VISIBLE_FACE and face_hits >= 2
            else 0.0
        )
        w_sum = max(1e-6, w_lex + w_voc + w_fac)
        w_lex, w_voc, w_fac = w_lex / w_sum, w_voc / w_sum, w_fac / w_sum

        combined = [
            2.35 * (w_lex * lex_raw[i] + w_voc * voc_raw[i] + w_fac * fac_raw[i])
            for i in range(3)
        ]
        # Hard lexical veto: if one story has clearly more hedges than the others,
        # never let weak face/prosody flip the pick away from it.
        hedge_counts = [len(HEDGING_WORDS.findall(s.transcript)) for s in stories]
        best_hedge_i = int(max(range(3), key=lambda i: hedge_counts[i]))
        if hedge_counts[best_hedge_i] >= 2 and hedge_counts[best_hedge_i] >= max(
            hedge_counts[j] for j in range(3) if j != best_hedge_i
        ) + 2:
            for i in range(3):
                if i == best_hedge_i:
                    combined[i] += 1.35
                else:
                    combined[i] -= 0.25
        probs_list = _softmax(combined)
        probs = {
            "story_1": probs_list[0],
            "story_2": probs_list[1],
            "story_3": probs_list[2],
        }
        best_idx = int(max(range(3), key=lambda i: probs_list[i]))
        chosen_num = best_idx + 1
        conf = probs_list[best_idx]

        sparse_visual = (
            modality_mode != ModalityMode.AUDIO_PLUS_VISIBLE_FACE
            or all(s.sparse_visual_evidence for s in stories)
            or len(usable_visible_frames) < 4
        )
        all_cited: list[str] = []
        for s in stories:
            for fid in s.cited_frame_ids:
                if fid not in all_cited:
                    all_cited.append(fid)
        if not all_cited and usable_visible_frames and modality_mode == ModalityMode.AUDIO_PLUS_VISIBLE_FACE:
            # Cite representative reference still frame so replay shows the straight-face observation honestly
            ref_frames = [f.frame_id for f in usable_visible_frames if f.crop_phase == "reference_still"]
            all_cited = ref_frames[:1] or [usable_visible_frames[len(usable_visible_frames) // 2].frame_id]

        observed_actions = [obs for s in stories for obs in s.observations]
        if not observed_actions and modality_mode == ModalityMode.AUDIO_PLUS_VISIBLE_FACE:
            if usable_visible_frames:
                observed_actions = ["Visible still face across stories; sparse facial motion evidence."]
            else:
                observed_actions = ["No usable speaker-attributed facial frames available."]

        rationale = sanitize_observable_text(
            " | ".join(per_story_notes),
            allow_micro_expression_claim=self.weights.allow_micro_expression_claim,
        )

        return GameGuessRecord(
            chosen_story_number=chosen_num,
            spoken_whisper=f"I think number {chosen_num} is the lie.",
            probabilities=probs,
            confidence=round(conf, 4),
            exploratory_disclosure=bool(self.weights.present_guess_as_exploratory or conf < 0.52),
            sparse_visual_disclosure=sparse_visual,
            cited_frame_ids=all_cited,
            observed_actions=observed_actions,
            comparison_rationale=rationale,
            locked_at_sec=round(stories[-1].end_sec, 3),
        )

    def _build_comparable_followup_questions(self, stories: Sequence[StorySpan]) -> list[str]:
        """Generates one comparable, neutral episodic verification question per story."""
        questions: list[str] = []
        for st in stories:
            snippet = " ".join(st.transcript.split()[:7]).strip(".,!?")
            questions.append(
                f"For story {st.story_number} ({snippet}...), what happened immediately beforehand and who else was there?"
            )
        return questions

    def _evaluate_deal_turn(
        self,
        utterances: Sequence[UtteranceSegment],
        events: Sequence[CandidateMotionInterval],
        frames: Sequence[FrameSample],
        modality_mode: ModalityMode,
    ) -> DealCueOutput:
        if modality_mode != ModalityMode.AUDIO_PLUS_VISIBLE_FACE:
            return DealCueOutput(
                action="silence",
                silence_reason="Deal cues require usable visible-face observations; sparse/masked visual mode defaults to silence.",
            )

        usable_frame_ids = {f.frame_id for f in frames if f.face_visible and f.usable and f.speaker_attributed}
        usable_events = [
            e
            for e in events
            if e.usable and e.speaker_attributed and e.apex_frame_id in usable_frame_ids
        ]
        if not usable_events:
            return DealCueOutput(
                action="silence",
                silence_reason="Sparse or unusable facial evidence on speaker; staying silent per Deal policy.",
            )

        # Match identifiable deal terms with temporal coincidence to usable facial events
        matched_pair: tuple[str, CandidateMotionInterval, float] | None = None
        for u in utterances:
            txt_len = max(1, len(u.text))
            for m in DEAL_TERM_PATTERN.finditer(u.text):
                term = m.group(0).strip()
                # Estimate term timestamp from word timestamps or proportional character offset across utterance
                rel_pos = m.start() / float(txt_len)
                term_sec = u.start_sec + rel_pos * max(0.0, u.end_sec - u.start_sec)
                for w in u.words:
                    if term.lower() in w.word.lower() or w.word.lower().strip(".,!?") in term.lower():
                        term_sec = w.start_sec
                        break
                for ev in usable_events:
                    signed_dt = term_sec - ev.apex_sec  # positive when facial cue precedes or coincides with term
                    dist = abs(signed_dt)
                    if -0.45 <= signed_dt <= 1.45:
                        base_conf = min(0.96, 0.58 + 1.45 * ev.peak_delta + 0.12 * min(1.0, u.pause_before_sec))
                        # Prefer concrete numbers/dates/deadlines and anticipatory coincidence right before the term
                        specificity_bonus = 0.06 if re.search(r"(\$|\d|monday|tuesday|wednesday|thursday|friday|q[1-4])", term, re.I) else 0.0
                        proximity_score = base_conf + specificity_bonus - 0.12 * dist
                        if matched_pair is None or proximity_score > matched_pair[2]:
                            matched_pair = (term, ev, proximity_score)

        if matched_pair is None:
            return DealCueOutput(
                action="silence",
                silence_reason="No usable facial observation coincided with an identifiable deal commitment.",
            )

        term, ev, _ = matched_pair
        conf = min(0.96, 0.58 + 1.45 * ev.peak_delta + 0.12 * max((u.pause_before_sec for u in utterances), default=0.0))
        if conf < self.weights.deal_cue_confidence_threshold:
            return DealCueOutput(
                action="silence",
                identified_term=term,
                confidence=round(conf, 4),
                silence_reason=(
                    f"Calibrated confidence ({conf:.2f}) below Deal threshold "
                    f"({self.weights.deal_cue_confidence_threshold:.2f})."
                ),
            )

        term_clean = term.strip(".,!?;:")
        if term_clean.lower() in ("friday", "monday", "tuesday", "wednesday", "thursday", "next week"):
            whisper = f"Ask what could delay {term_clean.capitalize() if term_clean.lower() != 'next week' else term_clean}."
        elif "$" in term_clean or "budget" in term_clean.lower() or "pricing" in term_clean.lower():
            whisper = f"Ask what constraint sits behind {term_clean}."
        else:
            whisper = f"Ask what condition affects {term_clean}."

        cited = []
        for fid in (ev.before_frame_id, ev.during_frame_id, ev.after_frame_id):
            if fid in usable_frame_ids and fid not in cited:
                cited.append(fid)

        return DealCueOutput(
            action="whisper_question",
            identified_term=term_clean,
            spoken_whisper=whisper,
            confidence=round(conf, 4),
            cited_frame_ids=cited,
            observable_evidence=sanitize_observable_text(
                ev.observable_action,
                self.weights.allow_micro_expression_claim,
            ),
        )

    def _evaluate_practice_turn(
        self,
        utterances: Sequence[UtteranceSegment],
        events: Sequence[CandidateMotionInterval],
        frames: Sequence[FrameSample],
        is_replay: bool,
    ) -> PracticeFeedbackOutput:
        joined = " ".join(u.text for u in utterances)
        exercise_done = bool(PRACTICE_END_PATTERN.search(joined)) or (is_replay and len(utterances) >= 2)
        if not exercise_done:
            return PracticeFeedbackOutput(exercise_complete=False)

        usable_frame_ids = {f.frame_id for f in frames if f.face_visible and f.usable and f.speaker_attributed}
        usable_events = [
            e for e in events if e.usable and e.speaker_attributed and e.apex_frame_id in usable_frame_ids
        ]

        if usable_events:
            top_ev = max(usable_events, key=lambda e: e.peak_delta)
            cited = [
                fid
                for fid in (top_ev.before_frame_id, top_ev.during_frame_id, top_ev.after_frame_id)
                if fid in usable_frame_ids
            ]
            target_phrase = f"'{top_ev.coinciding_word}'" if top_ev.coinciding_word else "that commitment"
            return PracticeFeedbackOutput(
                exercise_complete=True,
                replay_start_sec=round(max(0.0, top_ev.start_sec - 0.5), 3),
                replay_end_sec=round(top_ev.end_sec + 0.6, 3),
                cited_frame_ids=cited,
                observable_moment_summary=sanitize_observable_text(
                    top_ev.observable_action,
                    self.weights.allow_micro_expression_claim,
                ),
                suggested_followup=f"State {target_phrase} cleanly in one sentence, then pause instead of adding qualifiers.",
                confidence=round(min(0.95, 0.65 + top_ev.peak_delta), 4),
            )

        # Fallback to acoustic hesitation moment if visual motion was completely steady
        hesitant_utt = max(utterances, key=lambda u: u.pause_before_sec, default=None)
        if hesitant_utt is not None:
            ref_fids = [f.frame_id for f in frames if f.usable][:1]
            return PracticeFeedbackOutput(
                exercise_complete=True,
                replay_start_sec=round(max(0.0, hesitant_utt.start_sec - 0.4), 3),
                replay_end_sec=round(hesitant_utt.end_sec, 3),
                cited_frame_ids=ref_fids,
                observable_moment_summary=(
                    f"Steady facial posture with {hesitant_utt.pause_before_sec:.2f}s vocal pause before '{hesitant_utt.text[:28]}...'"
                ),
                suggested_followup="Keep your steady expression and tighten the pause before stating your core term.",
                confidence=0.74,
            )

        return PracticeFeedbackOutput(exercise_complete=True, confidence=0.5)

    def _enhance_with_live_gemini(
        self,
        activity: ActivityKind,
        utterances: Sequence[UtteranceSegment],
        frames: Sequence[FrameSample],
        events: Sequence[CandidateMotionInterval],
        stories: list[StorySpan],
        local_guess: GameGuessRecord | None,
        local_deal: DealCueOutput | None,
        local_practice: PracticeFeedbackOutput | None,
        modality_mode: ModalityMode,
    ) -> tuple[GameGuessRecord | None, DealCueOutput | None, PracticeFeedbackOutput | None, dict[str, Any]]:
        """Executes live Gemini 4 Neon + Gemini 4 Carbon + Gemini Decision 1 (`v1beta/decisions`)."""
        from google import genai

        client_v1 = genai.Client(
            api_key=self.settings.gemini_api_key,
            http_options={"timeout": int(self.settings.network_timeout_seconds * 1000)},
        )
        client_beta = genai.Client(
            api_key=self.settings.gemini_api_key,
            http_options={
                "api_version": "v1beta",
                "timeout": int(self.settings.network_timeout_seconds * 1000),
            },
        )

        usable_frame_lookup = {
            f.frame_id: f
            for f in frames
            if f.face_visible and f.usable and f.speaker_attributed
        }

        # Build multimodal input blocks for Gemini 4 Neon (Stage 1)
        neon_input: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "inferred_activity": activity.value,
                        "modality_mode": modality_mode.value,
                        "utterances": [
                            {
                                "start_sec": u.start_sec,
                                "end_sec": u.end_sec,
                                "speaker": u.speaker_id,
                                "text": u.text,
                                "pause_before_sec": u.pause_before_sec,
                                "mean_pitch_hz": u.mean_pitch_hz,
                                "speech_rate_wps": u.speech_rate_wps,
                            }
                            for u in utterances
                        ],
                        "tracker_events": [e.model_dump() for e in events if e.usable],
                    }
                )
                + "\n",
            }
        ]
        if modality_mode == ModalityMode.AUDIO_PLUS_VISIBLE_FACE:
            attached_crops = 0
            for fid, f_sample in usable_frame_lookup.items():
                if f_sample.jpeg_base64 and f_sample.crop_phase in ("before", "during", "after", "reference_still", "sampled"):
                    neon_input.append(
                        {"type": "text", "text": f"Frame {fid} ({f_sample.timestamp_sec:.2f}s, {f_sample.crop_phase}):\n"}
                    )
                    neon_input.append(
                        {
                            "type": "image",
                            "data": f_sample.jpeg_base64,
                            "mime_type": "image/jpeg",
                            "resolution": "low",
                        }
                    )
                    attached_crops += 1
                    if attached_crops >= 9:
                        break

        neon_schema = {
            "type": "object",
            "properties": {
                "activity": {
                    "type": "string",
                    "enum": ["two_truths_and_a_lie", "deal", "practice", "unclear"],
                },
                "visual_observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "frame_id": {"type": "string"},
                            "timestamp_sec": {"type": "number"},
                            "observable_change": {"type": "string"},
                        },
                        "required": ["frame_id", "observable_change"],
                    },
                },
            },
            "required": ["activity", "visual_observations"],
        }

        neon_interaction = client_v1.interactions.create(
            model=self.settings.neon_model,
            system_instruction=(
                "You are Sherlocked Stage 1 (Context & Visual Observation Extractor). "
                "Describe strictly physical, observable facial actions linked to exact frame_ids and timestamps "
                "(e.g., 'brief lip press before Friday'). Never use clinical or moralizing labels."
            ),
            input=neon_input,
            generation_config={"thinking_level": "low"},
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": neon_schema,
            },
            store=True,
        )
        neon_data = json.loads(neon_interaction.output_text or "{}")

        # Stage 2: Gemini 4 Carbon comparison & candidate phrasing
        carbon_schema = {
            "type": "object",
            "properties": {
                "comparison_summary": {"type": "string"},
                "candidate_deal_question": {"type": "string"},
                "candidate_practice_followup": {"type": "string"},
                "cited_frame_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["comparison_summary", "cited_frame_ids"],
        }
        carbon_interaction = client_v1.interactions.create(
            model=self.settings.carbon_model,
            previous_interaction_id=neon_interaction.id,
            system_instruction=(
                "You are Sherlocked Stage 2 (Comparative Analyst). "
                "Compare stories or deal/practice candidates using wording, prosody, and physical observations "
                f"(facial weight={self.weights.facial_evidence_weight:.2f}). Keep observations distinct from intent."
            ),
            input="Synthesize structured cross-evidence comparison and concise follow-up candidate if warranted.",
            generation_config={"thinking_level": "medium"},
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": carbon_schema,
            },
        )
        carbon_data = json.loads(carbon_interaction.output_text or "{}")

        # Stage 3: Gemini Decisions API (`gemini-decision-1`) final calibrated selection!
        decision_context = json.dumps(
            {
                "activity": activity.value,
                "stories": [s.model_dump() for s in stories],
                "neon_observations": neon_data.get("visual_observations", []),
                "carbon_comparison": carbon_data,
            }
        )
        questions: dict[str, Any] = {
            "visual_evidence_density": {
                "type": "score",
                "instructions": "Rate the density of usable speaker-attributed facial movement evidence.",
                "criteria": [
                    "No usable visual frames or masked/audio-only input.",
                    "Visible still face with minimal or zero target movement.",
                    "Moderate observable movement linked to specific timestamps.",
                    "Clear, well-timed observable facial movement linked to key words.",
                ],
            }
        }
        if activity == ActivityKind.GAME and len(stories) == 3:
            questions["is_round_playable"] = {
                "type": "noul",
                "instructions": "Are all three stories from the player intelligible and distinct enough to choose one lie?",
            }
            questions["lie_choice"] = {
                "type": "choice",
                "instructions": "Which of the three stories is most likely the fabricated lie based on the complete exchange?",
                "criteria": {
                    "story_1": "Story 1 is the fabricated story.",
                    "story_2": "Story 2 is the fabricated story.",
                    "story_3": "Story 3 is the fabricated story.",
                },
            }
        elif activity == ActivityKind.DEAL:
            questions["should_whisper_deal_cue"] = {
                "type": "noul",
                "instructions": "Does a usable speaker-attributed facial observation coincide with a specific deal term and justify whispering a concise follow-up question?",
                "criteria": {
                    "true_description": "Specific commitment/term coincides with usable speaker facial movement and supports a brief actionable question.",
                    "false_description": "Ordinary talk, honest steady delivery, or sparse/unusable facial evidence.",
                },
            }
        elif activity == ActivityKind.PRACTICE:
            questions["has_replayable_moment"] = {
                "type": "noul",
                "instructions": "Is the practice exercise ready for post-exercise replay feedback?",
            }

        dec_res = client_beta.decisions.create(
            model=self.settings.decision_model,
            input=[{"type": "text", "text": decision_context}],
            questions=questions,
        )
        answers = dec_res.answers
        valid_cited = [
            fid for fid in carbon_data.get("cited_frame_ids", []) if fid in usable_frame_lookup
        ]

        if activity == ActivityKind.GAME and local_guess is not None and "lie_choice" in answers:
            choice_ans = answers["lie_choice"]
            chosen_key = choice_ans.choice  # "story_1" | "story_2" | "story_3"
            chosen_num = int(chosen_key.split("_")[1])
            probs = {k: round(float(v), 4) for k, v in choice_ans.probabilities.items()}
            vis_score = float(answers["visual_evidence_density"].score)
            local_guess = local_guess.model_copy(
                update={
                    "chosen_story_number": chosen_num,
                    "spoken_whisper": f"I think number {chosen_num} is the lie.",
                    "probabilities": probs,
                    "confidence": round(float(choice_ans.confidence), 4),
                    "sparse_visual_disclosure": vis_score < 1.35,
                    "cited_frame_ids": valid_cited or local_guess.cited_frame_ids,
                    "comparison_rationale": sanitize_observable_text(
                        carbon_data.get("comparison_summary", local_guess.comparison_rationale),
                        self.weights.allow_micro_expression_claim,
                    ),
                }
            )
        elif activity == ActivityKind.DEAL and local_deal is not None and "should_whisper_deal_cue" in answers:
            noul_ans = answers["should_whisper_deal_cue"]
            p_yes = float(noul_ans.noul)
            if p_yes >= self.weights.deal_cue_confidence_threshold and local_deal.cited_frame_ids:
                q_txt = carbon_data.get("candidate_deal_question", "").strip() or local_deal.spoken_whisper
                local_deal = local_deal.model_copy(
                    update={
                        "action": "whisper_question",
                        "spoken_whisper": sanitize_observable_text(q_txt or "Ask what could delay that commitment."),
                        "confidence": round(p_yes, 4),
                        "cited_frame_ids": valid_cited or local_deal.cited_frame_ids,
                    }
                )
            else:
                local_deal = DealCueOutput(
                    action="silence",
                    identified_term=local_deal.identified_term,
                    confidence=round(p_yes, 4),
                    silence_reason="Gemini Decisions API calibrated noul below Deal whisper threshold.",
                )
        elif activity == ActivityKind.PRACTICE and local_practice is not None and local_practice.exercise_complete:
            q_txt = carbon_data.get("candidate_practice_followup", "").strip()
            if q_txt:
                local_practice = local_practice.model_copy(
                    update={
                        "suggested_followup": sanitize_observable_text(q_txt),
                        "cited_frame_ids": valid_cited or local_practice.cited_frame_ids,
                    }
                )

        meta = {
            "neon_interaction_id": neon_interaction.id,
            "carbon_interaction_id": carbon_interaction.id,
            "decisions_output_tokens": getattr(dec_res.usage, "total_output_tokens", 0),
        }
        return local_guess, local_deal, local_practice, meta

    def analyze_session_turn(self, req: SessionAnalyzeRequest) -> SessionAnalyzeResponse:
        # 1. Strictly enforce the Reveal Firewall before touching any guesser logic
        req.assert_reveal_firewall()

        # 2. Extract / enrich live utterances, frames, and quality failure flags
        utterances, frames, events, discarded_flags, effective_fps = self._prepare_media_inputs(req)
        # Repair common glasses-mic ASR misses so Game detection + lie scoring still fire.
        utterances = _normalize_utterances_asr(utterances)

        # 3. Strip any spoken reveal from guesser inputs while recording post-reveal transitions
        pre_reveal_utts, spoken_reveal_num, post_reveal_utts = _strip_reveal_utterances(utterances)

        session = self.store.get_session(req.session_id)
        activity, epoch, active_utts = self._detect_activity_and_epoch(
            session=session,
            pre_reveal_utts=pre_reveal_utts,
            revealed_num=spoken_reveal_num,
            post_reveal_utts=post_reveal_utts,
        )

        result_id = self._compute_result_id(req, epoch, activity, active_utts)
        cached = self.store.get_delivered_result(result_id)
        if cached is not None:
            return cached

        execution_mode = "live_gemini" if self.settings.has_live_gemini else "local_multimodal_engine"
        models_used = {
            "stage1_context_visual": self.settings.neon_model if self.settings.has_live_gemini else "local_neon_compatible_extractor",
            "stage2_comparison": self.settings.carbon_model if self.settings.has_live_gemini else "local_carbon_within_speaker_comparator",
            "stage3_decisions": self.settings.decision_model if self.settings.has_live_gemini else "local_calibrated_bayesian_decision",
        }

        status_label = "Listening for context"
        spoken_cue: str | None = None
        retry_requested = False
        retry_reason: str | None = None
        deal_output: DealCueOutput | None = None
        practice_output: PracticeFeedbackOutput | None = None
        live_meta: dict[str, Any] = {}

        if activity == ActivityKind.GAME:
            stories = (
                session.manual_stories_override
                if session.manual_stories_override is not None
                else self._segment_three_stories(pre_reveal_utts, events, frames)
            )
            session.stories = stories
            complete_stories = [s for s in stories if s.status == "complete"]

            if len(stories) >= 3 and len(complete_stories) < 3:
                status_label = "Game · Waiting for clear story completion"
                if any(u.overlapping_speech for u in pre_reveal_utts):
                    retry_requested = True
                    retry_reason = "Overlapping or incomplete speech left a story pending; please repeat the unclear story."
            elif len(complete_stories) < 3:
                status_label = f"Two Truths & a Lie · Tracking stories ({len(complete_stories)}/3)"
            else:
                # 3 complete stories from the same player!
                followup_asked = req.request_followup_questions or any(
                    FOLLOWUP_REQUEST_PATTERN.search(u.text) for u in pre_reveal_utts
                )
                total_words = sum(len(s.transcript.split()) for s in complete_stories[:3])
                solid_round = total_words >= 20 and all(
                    len(s.transcript.split()) >= 6 for s in complete_stories[:3]
                )
                if not solid_round:
                    status_label = "Two Truths & a Lie · Waiting for fuller story detail"
                else:
                    candidate_guess = self._score_stories_multimodal(
                        stories=complete_stories[:3],
                        utterances=pre_reveal_utts,
                        events=events,
                        frames=frames,
                        modality_mode=req.modality_mode,
                        is_followup=False,
                    )
                    # Live play needs a fast private whisper — skip blocking Gemini on live turns.
                    # Evaluation / replay still get the full live-LLM enhance path.
                    if self.settings.has_live_gemini and (req.is_evaluation or req.is_replay):
                        try:
                            candidate_guess, _, _, live_meta = self._enhance_with_live_gemini(
                                activity=activity,
                                utterances=pre_reveal_utts,
                                frames=frames,
                                events=events,
                                stories=complete_stories[:3],
                                local_guess=candidate_guess,
                                local_deal=None,
                                local_practice=None,
                                modality_mode=req.modality_mode,
                            )
                        except Exception as exc:
                            execution_mode = "local_multimodal_engine"
                            live_meta = {"gemini_fallback_error": str(exc)}
                    else:
                        execution_mode = "local_multimodal_engine"
                        live_meta = {"live_fast_path": True}
                    assert candidate_guess is not None

                    should_lock = session.initial_guess is None
                    # Correct a weak early lock (often Story #1 from a phone-glance face blip)
                    # when the fuller transcript clearly points elsewhere.
                    if (
                        session.initial_guess is not None
                        and candidate_guess.chosen_story_number != session.initial_guess.chosen_story_number
                        and (
                            session.initial_guess.sparse_visual_disclosure
                            or session.initial_guess.confidence < 0.62
                            or candidate_guess.confidence >= session.initial_guess.confidence + 0.12
                        )
                        and candidate_guess.confidence >= 0.55
                    ):
                        should_lock = True

                    if should_lock:
                        prev = session.initial_guess.chosen_story_number if session.initial_guess else None
                        session.initial_guess = candidate_guess
                        spoken_cue = candidate_guess.spoken_whisper
                        if prev is not None and prev != candidate_guess.chosen_story_number:
                            spoken_cue = (
                                f"Updating — I think number {candidate_guess.chosen_story_number} "
                                f"is the lie."
                            )
                        status_label = f"Game · Initial guess locked (#{candidate_guess.chosen_story_number})"
                    else:
                        status_label = (
                            f"Game · Initial guess locked (#{session.initial_guess.chosen_story_number})"
                        )

                if followup_asked and not session.followup_questions:
                    session.followup_questions = self._build_comparable_followup_questions(complete_stories[:3])
                    status_label = "Game · Follow-up questions ready"

                if (req.is_followup_round or (followup_asked and len(pre_reveal_utts) > 4)) and session.initial_guess is not None:
                    upd = self._score_stories_multimodal(
                        stories=complete_stories[:3],
                        utterances=pre_reveal_utts,
                        events=events,
                        frames=frames,
                        modality_mode=req.modality_mode,
                        is_followup=True,
                    )
                    session.updated_guess = upd
                    status_label = (
                        f"Game · Initial #{session.initial_guess.chosen_story_number} (locked) · "
                        f"Updated #{upd.chosen_story_number}"
                    )
                    if req.is_followup_round and spoken_cue is None:
                        spoken_cue = f"After follow-up, I think number {upd.chosen_story_number} is the lie."

                if session.game_revealed and session.revealed_lie_story_number is not None:
                    status_label = (
                        f"Game complete · Initial #{session.initial_guess.chosen_story_number} vs "
                        f"Reveal #{session.revealed_lie_story_number}"
                    )

        elif activity == ActivityKind.DEAL:
            deal_output = self._evaluate_deal_turn(
                utterances=active_utts,
                events=events,
                frames=frames,
                modality_mode=req.modality_mode,
            )
            # Live Gemini enhance only for evaluation/replay (keep live wearables fast).
            if self.settings.has_live_gemini and (req.is_evaluation or req.is_replay):
                try:
                    _, deal_output, _, live_meta = self._enhance_with_live_gemini(
                        activity=activity,
                        utterances=active_utts,
                        frames=frames,
                        events=events,
                        stories=[],
                        local_guess=None,
                        local_deal=deal_output,
                        local_practice=None,
                        modality_mode=req.modality_mode,
                    )
                except Exception as exc:
                    execution_mode = "local_multimodal_engine"
                    live_meta = {"gemini_fallback_error": str(exc)}
            assert deal_output is not None
            if deal_output.action == "whisper_question" and deal_output.spoken_whisper:
                spoken_cue = deal_output.spoken_whisper
                status_label = "Deal discussion · Follow-up cue ready"
            else:
                status_label = "Deal discussion · Monitoring quietly"

        elif activity == ActivityKind.PRACTICE:
            practice_output = self._evaluate_practice_turn(
                utterances=active_utts,
                events=events,
                frames=frames,
                is_replay=req.is_replay,
            )
            if self.settings.has_live_gemini and practice_output.exercise_complete and (
                req.is_evaluation or req.is_replay
            ):
                try:
                    _, _, practice_output, live_meta = self._enhance_with_live_gemini(
                        activity=activity,
                        utterances=active_utts,
                        frames=frames,
                        events=events,
                        stories=[],
                        local_guess=None,
                        local_deal=None,
                        local_practice=practice_output,
                        modality_mode=req.modality_mode,
                    )
                except Exception as exc:
                    execution_mode = "local_multimodal_engine"
                    live_meta = {"gemini_fallback_error": str(exc)}
            assert practice_output is not None
            if practice_output.exercise_complete and practice_output.suggested_followup:
                spoken_cue = practice_output.suggested_followup
                status_label = "Practice · Replay & coaching ready"
            else:
                status_label = (
                    "Practice · Replay & coaching ready"
                    if practice_output.exercise_complete
                    else "Practice · Recording rehearsal quietly"
                )

        else:
            status_label = "Listening for context"

        # Audio vs. Visual cue routing rules:
        # 1. Replay and evaluation calls are silent by default.
        # 2. If private route is active, whisper privately.
        # 3. If private playback route disappears, present the cue on screen instead!
        silent_mode = req.is_replay or req.is_evaluation
        should_speak = False
        present_on_screen = False
        if spoken_cue and not silent_mode:
            if req.audio_route.is_private:
                should_speak = True
                present_on_screen = False
            else:
                should_speak = False
                present_on_screen = True

        self.store.save_session(session)

        response = SessionAnalyzeResponse(
            result_id=result_id,
            session_id=req.session_id,
            activity_epoch=epoch,
            activity=activity,
            status_label=status_label,
            execution_mode=execution_mode,  # type: ignore[arg-type]
            models_used=models_used,
            already_delivered=False,
            should_speak_privately=should_speak,
            present_cue_on_screen=present_on_screen,
            spoken_cue_text=spoken_cue,
            retry_requested=retry_requested,
            retry_reason=retry_reason,
            stories=session.stories if activity == ActivityKind.GAME else [],
            initial_guess=session.initial_guess if activity == ActivityKind.GAME else None,
            followup_questions=session.followup_questions if activity == ActivityKind.GAME else [],
            updated_guess=session.updated_guess if activity == ActivityKind.GAME else None,
            revealed_lie_story_number=session.revealed_lie_story_number if activity == ActivityKind.GAME else None,
            deal_output=deal_output,
            practice_output=practice_output,
            candidate_events=[e for e in events if e.usable],
            discarded_quality_flags=discarded_flags,
            effective_fps=effective_fps,
            ModalityUsed=req.modality_mode,
            telemetry={
                "learned_facial_weight": self.weights.facial_evidence_weight,
                "exploratory_mode": self.weights.present_guess_as_exploratory,
                "scientific_status": self.weights.scientific_status,
                "usable_frames": sum(1 for f in frames if f.face_visible and f.usable and f.speaker_attributed),
                "total_frames": len(frames),
                **live_meta,
            },
        )
        self.store.record_delivered_result(response)
        return response

    def apply_manual_correction(
        self,
        session_id: str,
        corrected_stories: list[StorySpan] | None = None,
        revealed_lie_story_number: int | None = None,
    ) -> SessionRecord:
        session = self.store.get_session(session_id)
        if corrected_stories is not None:
            session.manual_stories_override = corrected_stories
            session.stories = corrected_stories
        if revealed_lie_story_number is not None:
            # Reveal stays outside guesser; only recorded for post-guess scoring/UI
            session.game_revealed = True
            session.revealed_lie_story_number = revealed_lie_story_number
        self.store.save_session(session)
        return session

    def run_managed_agent_debrief(self, session_id: str) -> dict[str, Any]:
        """Runs a post-session replay & weight-calibration debrief via Gemini Managed Agents (`antigravity-preview-09-2026` + `gemini-4-argon`).

        Keeps heavy remote sandboxed Python statistical analysis off the sub-second live whisper loop,
        while producing post-round calibration metrics, Brier scores, and coaching takeaways.
        """
        session = self.store.get_session(session_id)
        benchmark_path = self.settings.db_path.parent.parent / "research" / "results" / "benchmark_report.json"
        benchmark_excerpt: dict[str, Any] = {}
        if benchmark_path.exists():
            try:
                raw_bench = json.loads(benchmark_path.read_text(encoding="utf-8"))
                benchmark_excerpt = {
                    "paired_camera_vs_audio": raw_bench.get("experiments_2_3_game_and_camera_isolation", {}).get(
                        "paired_camera_vs_audio", {}
                    ),
                    "learned_pipeline_weights": raw_bench.get("learned_pipeline_weights", {}),
                }
            except Exception:
                benchmark_excerpt = {}

        session_payload = {
            "session_id": session.session_id,
            "activity": session.current_activity.value,
            "stories": [s.model_dump() for s in session.stories],
            "initial_guess": session.initial_guess.model_dump() if session.initial_guess else None,
            "updated_guess": session.updated_guess.model_dump() if session.updated_guess else None,
            "revealed_lie_story_number": session.revealed_lie_story_number,
            "benchmark_context": benchmark_excerpt,
        }

        if self.settings.has_live_gemini:
            try:
                from google import genai

                client_beta = genai.Client(
                    api_key=self.settings.gemini_api_key,
                    http_options={"api_version": "v1beta", "timeout": 120_000},
                )
                prompt = (
                    "You are Sherlocked's post-session calibration & coaching analyst running inside a Gemini Managed Agent sandbox. "
                    "Using Python in your remote sandbox, compute the Brier score (if revealed_lie_story_number is present), "
                    "audit whether visual observations vs. vocal prosody drove the locked guess, verify zero clinical/moralizing "
                    "labels appear, and output a concise Markdown coaching & calibration summary.\n\n"
                    f"Session JSON:\n{json.dumps(session_payload, indent=2)}"
                )
                interaction = client_beta.interactions.create(
                    agent=self.settings.managed_agent_id,
                    input=prompt,
                    environment="remote",
                    agent_config={
                        "type": "antigravity",
                        "model": self.settings.managed_agent_model,
                    },
                )
                return {
                    "session_id": session_id,
                    "execution_mode": "live_managed_agent",
                    "agent": self.settings.managed_agent_id,
                    "model": self.settings.managed_agent_model,
                    "environment_id": getattr(interaction, "environment_id", None),
                    "debrief_markdown": sanitize_observable_text(
                        interaction.output_text or "",
                        allow_micro_expression_claim=self.weights.allow_micro_expression_claim,
                    ),
                }
            except Exception as exc:
                fallback_note = str(exc)
            else:
                fallback_note = None
        else:
            fallback_note = "GEMINI_API_KEY not configured; computed deterministic local statistical debrief."

        # Deterministic local fallback debrief (computes exact multi-class Brier score when reveal is known)
        brier_score: float | None = None
        if session.initial_guess and session.revealed_lie_story_number in (1, 2, 3):
            probs = session.initial_guess.probabilities
            brier_score = round(
                sum(
                    (probs.get(f"story_{idx}", 0.0) - (1.0 if idx == session.revealed_lie_story_number else 0.0)) ** 2
                    for idx in (1, 2, 3)
                ),
                4,
            )

        return {
            "session_id": session_id,
            "execution_mode": "local_statistical_debrief",
            "agent": self.settings.managed_agent_id,
            "model": self.settings.managed_agent_model,
            "brier_score": brier_score,
            "note": fallback_note,
            "debrief_markdown": (
                f"### Post-Session Calibration Debrief (`{session_id}`)\n"
                f"- **Activity**: `{session.current_activity.value}`\n"
                f"- **Locked Initial Guess**: `{session.initial_guess.chosen_story_number if session.initial_guess else 'N/A'}`\n"
                f"- **Revealed Ground Truth**: `{session.revealed_lie_story_number or 'Unrevealed'}`\n"
                f"- **Multi-Class Brier Score**: `{brier_score if brier_score is not None else 'Pending reveal'}`\n"
                f"- **Active Modality Weights**: Lexical `{self.weights.lexical_consistency_weight:.3f}`, "
                f"Prosody `{self.weights.vocal_prosody_weight:.3f}`, Facial `{self.weights.facial_evidence_weight:.3f}`"
            ),
        }
