"""ElevenLabs Scribe realtime + batch STT for live glasses/phone capture."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, AsyncIterator
from urllib.parse import urlencode

import httpx

logger = logging.getLogger(__name__)

ELEVEN_STT_HTTP = "https://api.elevenlabs.io/v1/speech-to-text"
ELEVEN_STT_WS = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

# Bias recognition toward Sherlocked game / deal vocabulary.
KEYTERMS = [
    "two truths and a lie",
    "two truths and one lie",
    "let's play two truths and a lie",
    "story one",
    "story two",
    "story three",
    "number one",
    "number two",
    "number three",
    "first",
    "second",
    "third",
    "firstly",
    "secondly",
    "thirdly",
    "the lie was",
    "the lie is",
    "follow-up questions",
    "ask me questions",
    "I once",
    "last summer",
]


def _ws_query() -> str:
    params: list[tuple[str, str]] = [
        ("model_id", "scribe_v2_realtime"),
        ("audio_format", "pcm_16000"),
        ("commit_strategy", "vad"),
        ("vad_silence_threshold_secs", "0.7"),
        ("language_code", "en"),
        ("include_timestamps", "true"),
    ]
    for term in KEYTERMS:
        params.append(("keyterms", term))
    return urlencode(params)


async def transcribe_pcm16_wav_bytes(api_key: str, wav_bytes: bytes) -> dict[str, Any]:
    """Batch STT for short PCM/WAV clips. Returns ElevenLabs JSON."""
    # Multipart fields for httpx async client.
    files: list[tuple[str, Any]] = [
        ("model_id", (None, "scribe_v2")),
        ("language_code", (None, "en")),
        ("timestamps_granularity", (None, "word")),
        ("tag_audio_events", (None, "false")),
        ("file", ("chunk.wav", wav_bytes, "audio/wav")),
    ]
    for term in KEYTERMS[:40]:
        files.append(("keyterms", (None, term)))

    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post(
            ELEVEN_STT_HTTP,
            headers={"xi-api-key": api_key},
            files=files,
        )
        resp.raise_for_status()
        return resp.json()


def utterances_from_eleven_response(
    payload: dict[str, Any],
    *,
    time_offset_sec: float = 0.0,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    """Map ElevenLabs word timestamps into Sherlocked utterance dicts."""
    text = (payload.get("text") or "").strip()
    words_raw = payload.get("words") or []
    words: list[dict[str, Any]] = []
    for w in words_raw:
        if (w.get("type") or "word") not in ("word", None):
            continue
        token = (w.get("text") or "").strip()
        if not token:
            continue
        start = float(w.get("start") or 0.0) + time_offset_sec
        end = float(w.get("end") or start + 0.12) + time_offset_sec
        words.append(
            {
                "word": token,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "confidence": float(w.get("logprob") or 0.9),
            }
        )
    if not text and words:
        text = " ".join(w["word"] for w in words)
    if not text:
        return []
    start_sec = words[0]["start_sec"] if words else time_offset_sec
    end_sec = words[-1]["end_sec"] if words else time_offset_sec + 0.4
    return [
        {
            "segment_id": f"utt_{start_index:03d}",
            "text": text,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "pause_before_sec": 0.0,
            "speech_rate_wps": round(len(words) / max(0.2, end_sec - start_sec), 3) if words else None,
            "words": words,
        }
    ]


async def proxy_elevenlabs_realtime(
    api_key: str,
    client_audio: AsyncIterator[bytes | dict[str, Any]],
    send_to_client,
) -> None:
    """
    Bridge: phone/glasses PCM (int16 LE mono 16k) <-> ElevenLabs realtime WS.
    `client_audio` yields either raw PCM bytes or control dicts ({"commit": true}).
    `send_to_client` is an async callable that accepts a JSON-serializable dict.
    """
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("websockets package required for ElevenLabs realtime") from exc

    url = f"{ELEVEN_STT_WS}?{_ws_query()}"
    headers = {"xi-api-key": api_key}

    async with websockets.connect(url, additional_headers=headers, max_size=8_000_000) as el_ws:
        await send_to_client({"type": "session", "status": "connected", "provider": "elevenlabs"})

        async def pump_up() -> None:
            async for item in client_audio:
                if isinstance(item, dict):
                    if item.get("commit"):
                        await el_ws.send(json.dumps({"message_type": "commit"}))
                    continue
                if not item:
                    continue
                b64 = base64.b64encode(item).decode("ascii")
                await el_ws.send(
                    json.dumps(
                        {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": b64,
                            "sample_rate": 16000,
                        }
                    )
                )

        async def pump_down() -> None:
            async for raw in el_ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                mtype = msg.get("message_type") or msg.get("type") or ""
                if mtype in ("partial_transcript",):
                    await send_to_client(
                        {
                            "type": "partial",
                            "text": msg.get("text") or "",
                        }
                    )
                elif mtype in ("committed_transcript", "committed_transcript_with_timestamps"):
                    await send_to_client(
                        {
                            "type": "committed",
                            "text": msg.get("text") or "",
                            "words": msg.get("words") or [],
                            "language_code": msg.get("language_code"),
                        }
                    )
                elif mtype == "session_started":
                    await send_to_client({"type": "session", "status": "started", "raw": msg})
                elif mtype in ("error", "auth_error"):
                    await send_to_client({"type": "error", "detail": msg})
                    break

        up = asyncio.create_task(pump_up())
        down = asyncio.create_task(pump_down())
        done, pending = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            exc = t.exception()
            if exc:
                logger.exception("ElevenLabs proxy task failed: %s", exc)
                await send_to_client({"type": "error", "detail": str(exc)})
