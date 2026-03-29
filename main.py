"""Music Analyzer v8.0 — Signal Engine v2 + Pipeline v3

Replaces: Gemini Pro/Flash, GPT-4o Audio, CLAP, M2D
Uses: Signal Engine v2 (Librosa + Essentia + MuQ) for all audio analysis
LLM: Claude Opus only — Prism (inference) + Muse (reverse prompting)

Endpoints:
  GET  /health         — healthcheck (Railway)
  POST /analyze        — single track analysis (webapp compat)
  POST /batch-analyze  — batch analysis from CDN URLs
  POST /compare        — A↔B comparison (Pipeline v3 Phase B)
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import httpx
import json
import os
import sys
import tempfile
import re
import numpy as np
import asyncio
import logging

# ── Signal Engine v2 import ──────────────────────────────────
_analyzer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio-analyzer")
if _analyzer_dir not in sys.path:
    sys.path.insert(0, _analyzer_dir)

from signal_engine import SignalEngine
from schemas import (
    TrackAnalysisV2, MusicInference, SunoPrompt,
    ComparisonResult, LibrosaFeatures, EssentiaFeatures, MuQFeatures,
)

# ── App ──────────────────────────────────────────────────────
app = FastAPI(title="Music Analyzer v8.0 — Signal Engine v2")
logger = logging.getLogger("analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ───────────────────────────────────────────────────
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jgfvwfalxnrdujaoqoiq.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")

MAX_CONCURRENT = 2
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
CLAUDE_TIMEOUT = 120  # seconds per LLM call

# ── Lazy singletons ─────────────────────────────────────────
_engine: SignalEngine | None = None
_claude: anthropic.AsyncAnthropic | None = None


def _get_engine() -> SignalEngine:
    global _engine
    if _engine is None:
        _engine = SignalEngine(enable_muq=True, enable_essentia=True)
        logger.info("Signal Engine v2 initialized")
    return _engine


def _get_claude() -> anthropic.AsyncAnthropic:
    global _claude
    if _claude is None:
        _claude = anthropic.AsyncAnthropic(
            api_key=ANTHROPIC_KEY,
            timeout=httpx.Timeout(CLAUDE_TIMEOUT, connect=10),
        )
    return _claude


# ════════════════════════════════════════════════════════════
# SUPABASE HELPER
# ════════════════════════════════════════════════════════════
async def save_to_supabase(data: dict) -> bool:
    """Insert analysis result into Supabase."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase not configured — skipping save")
        return False
    try:
        url = f"{SUPABASE_URL}/rest/v1/analysis_results"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, json=data, headers=headers)
            if r.status_code not in (200, 201):
                logger.warning(f"Supabase insert failed: {r.status_code} {r.text[:200]}")
                return False
        return True
    except Exception as e:
        logger.warning(f"Supabase save error: {e}")
        return False


# ════════════════════════════════════════════════════════════
# PRISM — Musical Inference (Claude Opus)
# ════════════════════════════════════════════════════════════
PRISM_PROMPT = """You are Prism, an expert music analyst. You receive objective signal analysis data from three independent backends (Librosa, Essentia, MuQ) and must infer musical characteristics.

SIGNAL ENGINE v2 ANALYSIS:
{signal_data}

{prompt_section}

TASK: Provide your musical inference based ONLY on the signal data above.
For genre: Cross-reference MuQ genre_top5 with Essentia/Librosa features. BPM + spectral profile should align with genre expectations.
For mood: Derive from energy, danceability, spectral characteristics, and key/scale.
For instruments: Infer from spectral centroid, bandwidth, and genre context.

{eval_section}

Respond ONLY in valid JSON:
{{
  "genre_primary": "most likely genre",
  "genre_secondary": ["other", "possible", "genres"],
  "mood": "mood description (2-4 words)",
  "instruments": ["detected", "instruments"],
  "vocal_type": "vocal style or 'instrumental'",
  "production_style": "brief production character",
  "energy_level": "low/medium/high",
  "confidence": 1-10,
  {eval_fields}
  "reasoning": "brief explanation of key inferences"
}}"""

EVAL_SECTION_A = """EVALUATION TASK: Also evaluate how accurately the original prompt was interpreted.
Compare the signal data against what the prompt intended.
For BPM: Compare Librosa BPM with prompt BPM (if specified). Handle half/double BPM.
For Key: Compare cross-validated key with prompt key (if specified). Handle enharmonic equivalents.
For Genre: Does MuQ genre_top5 match the prompt's intended genre?
For prompt_fidelity: Use the MuQ-MuLan score as primary metric."""

EVAL_FIELDS_A = """"genre_accuracy": 1-10,
  "bpm_accuracy": 1-10,
  "key_accuracy": 1-10,
  "instrument_accuracy": 1-10,
  "mood_accuracy": 1-10,
  "overall_score": 1.0-10.0,
  "prompt_conflicts": [{"conflict": "desc", "winner": "which won", "reason": "why"}],
  "token_feedback": [{"token": "word", "effectiveness": "high/medium/low", "reason": "why"}],
  "summary": "brief evaluation summary","""


async def run_prism(analysis: TrackAnalysisV2, prompt_text: str | None = None) -> dict:
    """Prism inference: TrackAnalysisV2 → MusicInference + optional evaluation."""
    client = _get_claude()
    signal_data = analysis.to_dict()

    if prompt_text:
        prompt_section = f"ORIGINAL PROMPT given to Suno:\n{prompt_text}"
        eval_section = EVAL_SECTION_A
        eval_fields = EVAL_FIELDS_A
    else:
        prompt_section = "No original prompt provided — pure inference mode."
        eval_section = ""
        eval_fields = ""

    try:
        message = await client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2500,
            messages=[{
                "role": "user",
                "content": PRISM_PROMPT.format(
                    signal_data=json.dumps(signal_data, indent=2, default=str),
                    prompt_section=prompt_section,
                    eval_section=eval_section,
                    eval_fields=eval_fields,
                ),
            }],
        )
        raw = _extract_json(message.content[0].text)
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "prism_parse_failed", "raw": raw[:500]}
    except anthropic.APITimeoutError:
        return {"error": "prism_timeout"}
    except anthropic.APIError as e:
        return {"error": f"prism_api_error: {e.status_code}"}


# ════════════════════════════════════════════════════════════
# MUSE — Reverse Prompting (Claude Opus)
# ════════════════════════════════════════════════════════════
MUSE_PROMPT = """You are Muse, a Suno.ai prompt engineering expert.
Given audio analysis data and musical inference, create the most effective Suno prompt
that would generate music matching these characteristics.

SIGNAL ENGINE v2 ANALYSIS:
{signal_data}

PRISM MUSICAL INFERENCE:
{inference_data}

RULES:
- Never use artist names in the prompt.
- Use tokens that Suno actually responds to: genre tags, mood descriptors, instrument mentions, BPM hints, key hints, production style words.
- Avoid contradictory tokens (e.g., don't combine 'soft acoustic' with 'heavy distortion').
- Keep the prompt concise (50-150 chars for foundation, 550-750 chars for performance layer).

Respond ONLY in valid JSON:
{{
  "foundation": "genre tags + BPM + key + core tokens",
  "performance": "detailed English prose describing the desired sound (550-750 chars)",
  "style_tags": ["individual", "style", "tokens"],
  "genre_tags": ["genre", "tags", "used"],
  "bpm_token": "BPM hint or null",
  "key_token": "key hint or null",
  "confidence": 1-10,
  "reasoning": "brief explanation of key prompt decisions"
}}"""


async def run_muse(analysis: TrackAnalysisV2, inference: dict) -> dict:
    """Muse reverse prompting: TrackAnalysisV2 + MusicInference → SunoPrompt."""
    client = _get_claude()

    try:
        message = await client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": MUSE_PROMPT.format(
                    signal_data=json.dumps(analysis.to_dict(), indent=2, default=str),
                    inference_data=json.dumps(inference, indent=2, default=str),
                ),
            }],
        )
        raw = _extract_json(message.content[0].text)
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "muse_parse_failed", "raw": raw[:500]}
    except anthropic.APITimeoutError:
        return {"error": "muse_timeout"}
    except anthropic.APIError as e:
        return {"error": f"muse_api_error: {e.status_code}"}


# ════════════════════════════════════════════════════════════
# DATA QUALITY SCORE — v3 (MuQ-based, no Gemini)
# ════════════════════════════════════════════════════════════
def compute_data_quality_score(analysis: TrackAnalysisV2) -> float:
    """Quality score (0-10) — 3 axes:
    Axis 1 (3.5): Librosa continuous features
    Axis 2 (3.5): Cross-engine validation (Librosa↔Essentia)
    Axis 3 (3.0): MuQ depth (embedding + genre + fidelity)
    """
    score = 0.0
    lib = analysis.librosa
    ess = analysis.essentia
    muq = analysis.muq
    xval = analysis.cross_validation

    # AXIS 1: Librosa continuous (max 3.5)
    if lib and not lib.error:
        if lib.bpm and lib.bpm > 0:
            score += 0.7
        if lib.duration and lib.duration > 120:
            score += 0.7
        elif lib.duration and lib.duration > 60:
            score += 0.42
        elif lib.duration and lib.duration > 30:
            score += 0.14
        if lib.dynamic_range and lib.dynamic_range > 15:
            score += 0.5
        elif lib.dynamic_range and lib.dynamic_range > 8:
            score += 0.3
        if lib.energy is not None:
            score += 0.6
        if lib.key:
            score += 1.0

    # AXIS 2: Cross-engine validation (max 3.5)
    if xval:
        if xval.bpm_match:
            score += 1.75
        elif xval.bpm_delta and abs(xval.bpm_delta) < 10:
            score += 0.875
        if xval.key_match:
            score += 1.75
        elif xval.energy_match:
            score += 0.5
    elif ess and not ess.error:
        # Fallback: essentia available but no cross-val computed
        score += 1.0

    # AXIS 3: MuQ depth (max 3.0)
    if muq and not muq.error:
        if muq.embedding_dim and muq.embedding_dim > 0:
            score += 1.0
        if muq.genre_top5 and len(muq.genre_top5) > 0:
            score += 1.0
            top_conf = muq.genre_top5[0][1] if muq.genre_top5[0] else 0
            if top_conf > 0.3:
                score += 0.5
        if muq.prompt_fidelity is not None and muq.prompt_fidelity > 0:
            score += 0.5

    return round(max(0.0, min(score, 10.0)), 1)


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════
def _extract_json(text: str) -> str:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


# ════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "8.0",
        "engine": "signal_engine_v2",
        "engine_initialized": _engine is not None,
        "claude_initialized": _claude is not None,
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    prompt_type: str = Form("style"),
    ip: str = Form(""),
    prompt_id: str = Form(""),
    analysis_mode: str = Form("full"),
):
    """Single track analysis — webapp compatible.

    analysis_mode:
      - "full": Signal Engine + Prism inference + evaluation (default)
      - "signal_only": Signal Engine only, no LLM
      - "reverse": Signal Engine + Prism + Muse reverse prompt
    """
    async with analysis_semaphore:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(400, "Empty audio file")

        # Write temp file for Signal Engine
        suffix = "." + (file.filename.rsplit(".", 1)[-1] if file.filename else "mp3")
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            engine = _get_engine()
            prompt_text = prompt.strip() or None

            # Stage 1: Signal Engine analysis
            track_analysis = engine.analyze(
                audio_path=tmp_path,
                track_id=prompt_id or None,
                prompt_text=prompt_text,
            )
            analysis_dict = track_analysis.to_dict()
            data_quality = compute_data_quality_score(track_analysis)

            result = {
                "version": "8.0",
                "analysis_mode": analysis_mode,
                "prompt_type": prompt_type,
                "track_analysis": analysis_dict,
                "data_quality_score": data_quality,
            }

            if analysis_mode == "signal_only":
                return result

            # Stage 2: Prism inference
            prism_result = await run_prism(track_analysis, prompt_text)
            result["prism_inference"] = prism_result

            # Stage 3 (optional): Muse reverse prompting
            if analysis_mode == "reverse" or not prompt_text:
                if not prism_result.get("error"):
                    muse_result = await run_muse(track_analysis, prism_result)
                    result["muse_reverse_prompt"] = muse_result
                else:
                    result["muse_reverse_prompt"] = {"error": "skipped_due_to_prism_error"}

            # Extract top-level scores for backward compat
            if prompt_text and "overall_score" in prism_result:
                result["overall_score"] = prism_result["overall_score"]
                result["genre_accuracy"] = prism_result.get("genre_accuracy")
                result["bpm_accuracy"] = prism_result.get("bpm_accuracy")
                result["instrument_accuracy"] = prism_result.get("instrument_accuracy")
                result["mood_accuracy"] = prism_result.get("mood_accuracy")
                result["summary"] = prism_result.get("summary", "")

            # Save to Supabase
            db_data = {
                "original_prompt": prompt_text or "",
                "gemini_report": json.dumps(analysis_dict, default=str),
                "genre_accuracy": prism_result.get("genre_accuracy"),
                "bpm_accuracy": prism_result.get("bpm_accuracy"),
                "instrument_accuracy": prism_result.get("instrument_accuracy"),
                "mood_accuracy": prism_result.get("mood_accuracy"),
                "overall_score": prism_result.get("overall_score"),
                "prompt_type": prompt_type,
                "analysis_mode": analysis_mode,
                "model_version": "v8_signal_engine_v2",
            }
            await save_to_supabase(db_data)

            return result

        except Exception as e:
            logger.exception("Analysis failed")
            raise HTTPException(500, f"Analysis failed: {str(e)}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ════════════════════════════════════════════════════════════
# BATCH ANALYZE
# ════════════════════════════════════════════════════════════
class BatchTrack(BaseModel):
    id: str
    audio_url: str
    style_prompt: str
    model_version: str = "v5"
    title: str = ""


class BatchRequest(BaseModel):
    tracks: list[BatchTrack]
    auth_token: str
    admin_key: str = ""
    analysis_mode: str = "full"


@app.post("/batch-analyze")
async def batch_analyze(req: BatchRequest):
    """Server-side batch: Railway downloads from CDN + runs analysis."""
    if ADMIN_KEY and req.admin_key != ADMIN_KEY:
        raise HTTPException(403, "Invalid admin_key")

    results = []
    sem = asyncio.Semaphore(1)  # Sequential for GPU-bound MuQ

    async def process_one(track: BatchTrack):
        async with sem:
            try:
                # Download audio
                async with httpx.AsyncClient(timeout=60) as dl:
                    r = await dl.get(
                        track.audio_url,
                        headers={"Authorization": req.auth_token},
                        follow_redirects=True,
                    )
                    if r.status_code != 200:
                        results.append({"id": track.id, "status": "dl_failed", "code": r.status_code})
                        return
                    audio_bytes = r.content

                # Write temp file
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name

                try:
                    engine = _get_engine()
                    track_analysis = engine.analyze(
                        audio_path=tmp_path,
                        track_id=track.id,
                        prompt_text=track.style_prompt,
                    )

                    prism_result = {}
                    if req.analysis_mode != "signal_only":
                        prism_result = await run_prism(track_analysis, track.style_prompt)

                    # Save to Supabase
                    db_data = {
                        "original_prompt": track.style_prompt,
                        "gemini_report": json.dumps(track_analysis.to_dict(), default=str),
                        "genre_accuracy": prism_result.get("genre_accuracy"),
                        "bpm_accuracy": prism_result.get("bpm_accuracy"),
                        "instrument_accuracy": prism_result.get("instrument_accuracy"),
                        "mood_accuracy": prism_result.get("mood_accuracy"),
                        "overall_score": prism_result.get("overall_score"),
                        "prompt_type": "style",
                        "analysis_mode": req.analysis_mode,
                        "model_version": track.model_version,
                    }
                    await save_to_supabase(db_data)

                    results.append({
                        "id": track.id,
                        "status": "ok",
                        "overall_score": prism_result.get("overall_score"),
                        "data_quality": compute_data_quality_score(track_analysis),
                    })
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            except Exception as e:
                results.append({"id": track.id, "status": "error", "error": str(e)})
            await asyncio.sleep(1)

    await asyncio.gather(*[process_one(t) for t in req.tracks])
    ok = sum(1 for r in results if r.get("status") == "ok")
    return {"total": len(req.tracks), "success": ok, "results": results}


# ════════════════════════════════════════════════════════════
# COMPARE — A↔B Pipeline v3 Phase B
# ════════════════════════════════════════════════════════════
class CompareRequest(BaseModel):
    track_a_path: str
    track_b_path: str
    prompt_text: str
    admin_key: str = ""


@app.post("/compare")
async def compare_tracks(req: CompareRequest):
    """Compare original (A) vs Suno-generated (B) track.
    Returns embedding cosine similarity + feature deltas.
    """
    if ADMIN_KEY and req.admin_key != ADMIN_KEY:
        raise HTTPException(403, "Invalid admin_key")

    engine = _get_engine()

    analysis_a = engine.analyze(req.track_a_path, track_id="A", prompt_text=req.prompt_text)
    analysis_b = engine.analyze(req.track_b_path, track_id="B", prompt_text=req.prompt_text)

    # Compute comparison metrics
    comparison = {}

    # 1. MuQ embedding cosine similarity
    if (analysis_a.muq and analysis_a.muq.embedding is not None
            and analysis_b.muq and analysis_b.muq.embedding is not None):
        emb_a = np.array(analysis_a.muq.embedding)
        emb_b = np.array(analysis_b.muq.embedding)
        norm = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        cosine = float(np.dot(emb_a, emb_b) / norm) if norm > 0 else 0.0
        comparison["embedding_cosine"] = round(cosine, 4)

    # 2. Genre rank shift
    if (analysis_a.muq and analysis_a.muq.genre_top5
            and analysis_b.muq and analysis_b.muq.genre_top5):
        genres_a = [g[0] for g in analysis_a.muq.genre_top5]
        genres_b = [g[0] for g in analysis_b.muq.genre_top5]
        comparison["genre_top1_a"] = genres_a[0] if genres_a else None
        comparison["genre_top1_b"] = genres_b[0] if genres_b else None
        comparison["genre_top1_match"] = (
            genres_a[0].lower() == genres_b[0].lower() if genres_a and genres_b else False
        )

    # 3. Feature deltas
    if analysis_a.librosa and analysis_b.librosa:
        la, lb = analysis_a.librosa, analysis_b.librosa
        comparison["bpm_delta"] = round((lb.bpm or 0) - (la.bpm or 0), 1)
        comparison["energy_delta"] = round((lb.energy or 0) - (la.energy or 0), 4)
        comparison["spectral_centroid_delta"] = round(
            (lb.spectral_centroid or 0) - (la.spectral_centroid or 0), 0
        )
        comparison["key_a"] = f"{la.key} {la.scale}" if la.key else None
        comparison["key_b"] = f"{lb.key} {lb.scale}" if lb.key else None
        comparison["key_match"] = (
            la.key == lb.key and la.scale == lb.scale if la.key and lb.key else False
        )

    # 4. Prompt fidelity delta
    if analysis_a.muq and analysis_b.muq:
        fid_a = analysis_a.muq.prompt_fidelity or 0
        fid_b = analysis_b.muq.prompt_fidelity or 0
        comparison["prompt_fidelity_a"] = round(fid_a, 4)
        comparison["prompt_fidelity_b"] = round(fid_b, 4)
        comparison["prompt_fidelity_delta"] = round(fid_b - fid_a, 4)

    return {
        "version": "8.0",
        "prompt_text": req.prompt_text,
        "comparison": comparison,
        "track_a": analysis_a.to_dict(),
        "track_b": analysis_b.to_dict(),
    }
