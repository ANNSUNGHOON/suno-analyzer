# Suno Analyzer — Signal Engine v2

Audio analysis backend for the Suno Prompter webapp.

## Stack
- FastAPI + Uvicorn (Railway)
- Signal Engine v2: Librosa + Essentia + MuQ
- Layer 2: Audio Perception (Gemini 2.5 Flash)
- LLM: Claude (Prism inference + Muse prompting)

## Endpoints
- `GET /health` — healthcheck
- `POST /analyze` — single track analysis
- `POST /batch-analyze` — batch from CDN URLs
- `POST /compare` — A/B comparison

## Deploy
Railway auto-deploys from `main` branch.
Requires env vars: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`
