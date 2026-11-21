#!/usr/bin/env bash
set -euo pipefail
# Render sets $PORT; keep a single worker so MediaPipe/Whisper stay in one process.
exec uvicorn backend.server:app --host 0.0.0.0 --port "${PORT:-8787}" --workers 1
