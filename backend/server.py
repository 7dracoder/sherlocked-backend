"""FastAPI server for Sherlocked keeping Gemini credentials server-side."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import FIXTURES_DIR, RESEARCH_DIR, LearnedWeights, get_settings
from .elevenlabs_asr import proxy_elevenlabs_realtime, transcribe_pcm16_wav_bytes, utterances_from_eleven_response
from .gemini_pipeline import SherlockedPipeline
from .schemas import (
    CaptureSourceKind,
    ManualCorrectionRequest,
    ModalityMode,
    SessionAnalyzeRequest,
    SessionAnalyzeResponse,
)
from .vision_tracker import render_face_masked_video

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sherlocked Multimodal Inference Server",
    version="1.0.0",
    description="Server-side Gemini 4 Neon, Gemini 4 Carbon, and Gemini Decisions API pipeline for Ray-Ban Meta & iPhone.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_PIPELINE: SherlockedPipeline | None = None


def get_pipeline() -> SherlockedPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = SherlockedPipeline()
    else:
        _PIPELINE.reload_weights()
    return _PIPELINE


@app.get("/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    weights = LearnedWeights.load()
    return {
        "status": "ok",
        "live_gemini_configured": settings.has_live_gemini,
        "elevenlabs_configured": settings.has_elevenlabs,
        "asr_provider": "elevenlabs" if settings.has_elevenlabs else "apple_or_whisper_fallback",
        "execution_mode": "live_gemini" if settings.has_live_gemini else "local_multimodal_engine",
        "models": {
            "neon": settings.neon_model,
            "carbon": settings.carbon_model,
            "decisions": settings.decision_model,
            "managed_agent": settings.managed_agent_id,
            "managed_agent_model": settings.managed_agent_model,
        },
        "learned_weights": {
            "facial_evidence_weight": weights.facial_evidence_weight,
            "vocal_prosody_weight": weights.vocal_prosody_weight,
            "lexical_consistency_weight": weights.lexical_consistency_weight,
            "present_guess_as_exploratory": weights.present_guess_as_exploratory,
            "scientific_status": weights.scientific_status,
        },
    }


class ASRChunkRequest(BaseModel):
    """Short WAV/PCM clip from the phone for ElevenLabs batch STT."""

    audio_wav_base64: str
    time_offset_sec: float = 0.0
    utterance_start_index: int = 1


@app.post("/api/v1/asr/transcribe")
async def asr_transcribe_chunk(req: ASRChunkRequest) -> dict[str, Any]:
    settings = get_settings()
    if not settings.has_elevenlabs:
        raise HTTPException(status_code=503, detail="ELEVENLABS_API_KEY is not configured on the server.")
    try:
        wav_bytes = base64.b64decode(req.audio_wav_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid audio_wav_base64") from exc
    if len(wav_bytes) < 200:
        raise HTTPException(status_code=400, detail="Audio clip too short.")
    try:
        payload = await transcribe_pcm16_wav_bytes(settings.elevenlabs_api_key, wav_bytes)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"ElevenLabs error: {exc.response.text[:400]}") from exc
    except Exception as exc:
        logger.exception("ElevenLabs batch STT failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    utterances = utterances_from_eleven_response(
        payload,
        time_offset_sec=req.time_offset_sec,
        start_index=req.utterance_start_index,
    )
    return {
        "provider": "elevenlabs",
        "text": payload.get("text") or "",
        "utterances": utterances,
        "raw_word_count": len(payload.get("words") or []),
    }


@app.websocket("/api/v1/asr/live")
async def asr_live_ws(websocket: WebSocket) -> None:
    """Phone/glasses PCM stream → ElevenLabs Scribe realtime → partial/committed transcripts."""
    await websocket.accept()
    settings = get_settings()
    if not settings.has_elevenlabs:
        await websocket.send_json({"type": "error", "detail": "ELEVENLABS_API_KEY is not configured."})
        await websocket.close(code=1013)
        return

    queue: asyncio.Queue[bytes | dict[str, Any] | None] = asyncio.Queue(maxsize=64)

    async def client_audio() -> Any:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    async def send_to_client(msg: dict[str, Any]) -> None:
        await websocket.send_json(msg)

    proxy_task = asyncio.create_task(
        proxy_elevenlabs_realtime(settings.elevenlabs_api_key, client_audio(), send_to_client)
    )
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"] is not None:
                try:
                    queue.put_nowait(message["bytes"])
                except asyncio.QueueFull:
                    _ = queue.get_nowait()
                    queue.put_nowait(message["bytes"])
            elif "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "commit" or payload.get("commit"):
                    await queue.put({"commit": True})
                elif payload.get("type") == "audio" and payload.get("audio_base_64"):
                    raw = base64.b64decode(payload["audio_base_64"])
                    await queue.put(raw)
                elif payload.get("type") == "end":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await queue.put(None)
        proxy_task.cancel()
        try:
            await proxy_task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("ASR live proxy ended with error")


@app.post("/api/v1/session/analyze", response_model=SessionAnalyzeResponse)
def analyze_session(req: SessionAnalyzeRequest) -> SessionAnalyzeResponse:
    pipeline = get_pipeline()
    try:
        return pipeline.analyze_session_turn(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/session/manual-correction")
def manual_correction(req: ManualCorrectionRequest) -> dict[str, Any]:
    pipeline = get_pipeline()
    record = pipeline.apply_manual_correction(
        session_id=req.session_id,
        corrected_stories=req.corrected_stories,
        revealed_lie_story_number=req.revealed_lie_story_number,
    )
    return {
        "status": "updated",
        "session_id": record.session_id,
        "game_revealed": record.game_revealed,
        "revealed_lie_story_number": record.revealed_lie_story_number,
        "initial_guess_locked": record.initial_guess.chosen_story_number if record.initial_guess else None,
        "updated_guess": record.updated_guess.chosen_story_number if record.updated_guess else None,
        "stories_count": len(record.stories),
    }


@app.post("/api/v1/session/{session_id}/reset")
def reset_session(session_id: str) -> dict[str, str]:
    pipeline = get_pipeline()
    pipeline.store.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@app.post("/api/v1/session/{session_id}/agent-debrief")
def session_managed_agent_debrief(session_id: str) -> dict[str, Any]:
    pipeline = get_pipeline()
    return pipeline.run_managed_agent_debrief(session_id)


@app.get("/api/v1/fixtures")
def list_fixtures() -> dict[str, Any]:
    manifest_path = FIXTURES_DIR / "manifest.json"
    if not manifest_path.exists():
        return {"fixtures": []}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Strip any ground truth fields from public fixture listing so UI/guesser remains blind until reveal
    sanitized = []
    for item in data.get("fixtures", []):
        clean_item = {
            k: v
            for k, v in item.items()
            if k not in ("actual_lie_story_number", "secret_constraint", "ground_truth_intervals")
        }
        sanitized.append(clean_item)
    return {"fixtures": sanitized, "is_synthetic_consented_fixture_set": True}


@app.get("/api/v1/fixtures/{fixture_id}/media/{variant}")
def get_fixture_media(fixture_id: str, variant: str = "original") -> FileResponse:
    manifest_path = FIXTURES_DIR / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Fixtures manifest not found.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    match = next((f for f in manifest.get("fixtures", []) if f["fixture_id"] == fixture_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Fixture {fixture_id} not found.")

    orig_path = FIXTURES_DIR / match["video_relpath"]
    if not orig_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing.")

    if variant == "masked":
        masked_path = orig_path.with_name(f"{orig_path.stem}_masked.mp4")
        if not masked_path.exists():
            render_face_masked_video(orig_path, masked_path)
        return FileResponse(masked_path, media_type="video/mp4")

    return FileResponse(orig_path, media_type="video/mp4")


@app.post("/api/v1/fixtures/{fixture_id}/replay", response_model=SessionAnalyzeResponse)
def replay_fixture(
    fixture_id: str,
    face_masked: bool = Query(default=False),
    include_followup: bool = Query(default=False),
) -> SessionAnalyzeResponse:
    manifest_path = FIXTURES_DIR / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Fixtures manifest not found.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    match = next((f for f in manifest.get("fixtures", []) if f["fixture_id"] == fixture_id), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Fixture {fixture_id} not found.")

    video_path = FIXTURES_DIR / match["video_relpath"]
    if face_masked:
        masked_path = video_path.with_name(f"{video_path.stem}_masked.mp4")
        if not masked_path.exists():
            render_face_masked_video(video_path, masked_path)
        video_path = masked_path

    pipeline = get_pipeline()
    session_id = f"replay_{fixture_id}_{'masked' if face_masked else 'visible'}"
    pipeline.store.reset_session(session_id)

    req = SessionAnalyzeRequest(
        session_id=session_id,
        turn_id="replay_run_1",
        capture_source=CaptureSourceKind.REPLAY,
        modality_mode=(
            ModalityMode.AUDIO_PLUS_FACE_MASKED_VIDEO
            if face_masked
            else ModalityMode.AUDIO_PLUS_VISIBLE_FACE
        ),
        is_replay=True,
        request_followup_questions=include_followup,
        is_followup_round=include_followup,
        audio_video_drift_ms=float(match.get("audio_video_drift_ms", 0.0)),
        video_mp4_base64=base64.b64encode(video_path.read_bytes()).decode("ascii"),
    )
    return pipeline.analyze_session_turn(req)


@app.get("/api/v1/research/summary")
def research_summary() -> dict[str, Any]:
    results_path = RESEARCH_DIR / "results" / "benchmark_report.json"
    if not results_path.exists():
        return {"status": "pending", "weights": LearnedWeights.load().__dict__}
    return json.loads(results_path.read_text(encoding="utf-8"))
