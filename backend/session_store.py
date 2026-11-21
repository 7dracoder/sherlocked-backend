"""SQLite-backed session state and idempotent result_id ledger for Sherlocked."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import (
    ActivityKind,
    GameGuessRecord,
    SessionAnalyzeResponse,
    StorySpan,
)


@dataclass
class SessionRecord:
    session_id: str
    activity_epoch: int = 1
    current_activity: ActivityKind = ActivityKind.UNCLEAR
    game_revealed: bool = False
    revealed_lie_story_number: int | None = None
    stories: list[StorySpan] = field(default_factory=list)
    initial_guess: GameGuessRecord | None = None
    followup_questions: list[str] = field(default_factory=list)
    updated_guess: GameGuessRecord | None = None
    manual_stories_override: list[StorySpan] | None = None


class SessionStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at REAL DEFAULT (unixepoch('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS delivered_results (
                    result_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at REAL DEFAULT (unixepoch('now'))
                )
                """
            )
            conn.commit()

    def get_session(self, session_id: str) -> SessionRecord:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return SessionRecord(session_id=session_id)
            raw: dict[str, Any] = json.loads(row["payload_json"])
            return SessionRecord(
                session_id=session_id,
                activity_epoch=int(raw.get("activity_epoch", 1)),
                current_activity=ActivityKind(raw.get("current_activity", ActivityKind.UNCLEAR.value)),
                game_revealed=bool(raw.get("game_revealed", False)),
                revealed_lie_story_number=raw.get("revealed_lie_story_number"),
                stories=[StorySpan.model_validate(s) for s in raw.get("stories", [])],
                initial_guess=(
                    GameGuessRecord.model_validate(raw["initial_guess"])
                    if raw.get("initial_guess")
                    else None
                ),
                followup_questions=list(raw.get("followup_questions", [])),
                updated_guess=(
                    GameGuessRecord.model_validate(raw["updated_guess"])
                    if raw.get("updated_guess")
                    else None
                ),
                manual_stories_override=(
                    [StorySpan.model_validate(s) for s in raw["manual_stories_override"]]
                    if raw.get("manual_stories_override") is not None
                    else None
                ),
            )

    def save_session(self, record: SessionRecord) -> None:
        payload = {
            "activity_epoch": record.activity_epoch,
            "current_activity": record.current_activity.value,
            "game_revealed": record.game_revealed,
            "revealed_lie_story_number": record.revealed_lie_story_number,
            "stories": [s.model_dump() for s in record.stories],
            "initial_guess": record.initial_guess.model_dump() if record.initial_guess else None,
            "followup_questions": record.followup_questions,
            "updated_guess": record.updated_guess.model_dump() if record.updated_guess else None,
            "manual_stories_override": (
                [s.model_dump() for s in record.manual_stories_override]
                if record.manual_stories_override is not None
                else None
            ),
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, payload_json, updated_at)
                VALUES (?, ?, unixepoch('now'))
                ON CONFLICT(session_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = unixepoch('now')
                """,
                (record.session_id, json.dumps(payload)),
            )
            conn.commit()

    def get_delivered_result(self, result_id: str) -> SessionAnalyzeResponse | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT response_json FROM delivered_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if not row:
                return None
            resp = SessionAnalyzeResponse.model_validate_json(row["response_json"])
            return resp.model_copy(
                update={
                    "already_delivered": True,
                    "should_speak_privately": False,
                    "present_cue_on_screen": False,
                }
            )

    def record_delivered_result(self, response: SessionAnalyzeResponse) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO delivered_results (result_id, session_id, response_json)
                VALUES (?, ?, ?)
                """,
                (response.result_id, response.session_id, response.model_dump_json()),
            )
            conn.commit()

    def reset_session(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM delivered_results WHERE session_id = ?", (session_id,))
            conn.commit()
