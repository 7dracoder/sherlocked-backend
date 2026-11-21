# Sherlocked Backend

Standalone FastAPI backend for the Sherlocked iOS app (ASR proxy, multimodal analyze, fixtures).

## Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY / ELEVENLABS_API_KEY
uvicorn backend.server:app --host 0.0.0.0 --port 8787
```

## Render

Connected via `render.yaml`. Health: `GET /health`.
