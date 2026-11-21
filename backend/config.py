"""Central configuration and learned weights loader for Sherlocked."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
WEIGHTS_PATH = ROOT_DIR / "config" / "learned_pipeline_weights.json"
FIXTURES_DIR = ROOT_DIR / "fixtures"
RESEARCH_DIR = ROOT_DIR / "research"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ENV_PATH)


@dataclass(frozen=True)
class LearnedWeights:
    facial_evidence_weight: float = 0.22
    vocal_prosody_weight: float = 0.38
    lexical_consistency_weight: float = 0.40
    deal_cue_confidence_threshold: float = 0.72
    game_min_playable_confidence: float = 0.55
    present_guess_as_exploratory: bool = True
    allow_micro_expression_claim: bool = False
    calibration_source: str = "uninitialized_default"
    camera_gain_over_audio: float = 0.0
    camera_gain_p_value: float = 1.0
    scientific_status: str = "pending"
    notes: str = "Default conservative priors (DePaulo et al.) prior to benchmark calibration."

    @classmethod
    def load(cls, path: Path = WEIGHTS_PATH) -> "LearnedWeights":
        if not path.exists():
            return cls()
        try:
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            allowed = {k: payload[k] for k in cls.__dataclass_fields__ if k in payload}
            return cls(**allowed)
        except Exception:
            return cls()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    neon_model: str = field(
        default_factory=lambda: os.environ.get("SHERLOCKED_NEON_MODEL", "gemini-4-neon")
    )
    carbon_model: str = field(
        default_factory=lambda: os.environ.get("SHERLOCKED_CARBON_MODEL", "gemini-4-carbon")
    )
    decision_model: str = field(
        default_factory=lambda: os.environ.get("SHERLOCKED_DECISION_MODEL", "gemini-decision-1")
    )
    managed_agent_id: str = field(
        default_factory=lambda: os.environ.get("SHERLOCKED_MANAGED_AGENT", "antigravity-preview-09-2026")
    )
    managed_agent_model: str = field(
        default_factory=lambda: os.environ.get("SHERLOCKED_ARGON_MODEL", "gemini-4-argon")
    )
    transcribe_model: str = field(
        default_factory=lambda: os.environ.get("SHERLOCKED_TRANSCRIBE_MODEL", "gemini-3.5-transcribe")
    )
    elevenlabs_api_key: str = field(
        default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY", "").strip()
    )
    host: str = field(default_factory=lambda: os.environ.get("SHERLOCKED_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("SHERLOCKED_PORT", "8787")))
    restricted_datasets_dir: Path | None = field(
        default_factory=lambda: (
            Path(os.environ["RESTRICTED_DATASETS_DIR"]).expanduser()
            if os.environ.get("RESTRICTED_DATASETS_DIR", "").strip()
            else None
        )
    )
    network_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("SHERLOCKED_TIMEOUT_SECONDS", "25.0"))
    )
    db_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("SHERLOCKED_DB_PATH", str(ROOT_DIR / "backend" / "sherlocked_state.sqlite3"))
        )
    )

    @property
    def has_live_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_elevenlabs(self) -> bool:
        return bool(self.elevenlabs_api_key)


def get_settings() -> Settings:
    _load_dotenv(ENV_PATH)
    return Settings()
