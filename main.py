from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import anthropic
from openai import OpenAI
import httpx
import json
import os
import tempfile
import re
import numpy as np
import asyncio
import base64

app = FastAPI(title="Suno Audio Analyzer v5 — Quad Engine + Mode B")

MAX_CONCURRENT = 2
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jgfvwfalxnrdujaoqoiq.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")  # Set this in Railway env vars

# Model config — easy to swap
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
GEMINI_FLASH_MODEL = "gemini-2.5-flash-preview-04-17"  # Lite tier

# Quality threshold for re-eval recommendation
REEVAL_SCORE_THRESHOLD = 6.0

genai.configure(api_key=GEMINI_KEY)


# ════════════════════════════════════════
# ENGINE 1: LIBROSA — Mathematical Analysis
# ════════════════════════════════════════
def analyze_with_librosa(audio_bytes: bytes) -> dict:
    """Mathematically precise: BPM, key, frequency distribution, dynamics."""
    import librosa
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # BPM
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)

        # Key via chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key_idx = int(np.argmax(chroma_mean))
        detected_key = key_names[detected_key_idx]
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        major_corr = float(np.corrcoef(chroma_mean, np.roll(major_profile, detected_key_idx))[0, 1])
        minor_corr = float(np.corrcoef(chroma_mean, np.roll(minor_profile, detected_key_idx))[0, 1])
        scale = "major" if major_corr > minor_corr else "minor"

        # Spectral
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

        # RMS / Dynamics
        rms = librosa.feature.rms(y=y)
        avg_rms = float(np.mean(rms))
        max_rms = float(np.max(rms))
        dynamic_range_db = float(20 * np.log10(max_rms / (avg_rms + 1e-10)))

        # Frequency bands
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        total_energy = float(np.sum(S ** 2)) + 1e-10
        low_pct = round(float(np.sum(S[freqs < 250] ** 2)) / total_energy * 100, 1)
        mid_pct = round(float(np.sum(S[(freqs >= 250) & (freqs < 4000)] ** 2)) / total_energy * 100, 1)
        high_pct = round(float(np.sum(S[freqs >= 4000] ** 2)) / total_energy * 100, 1)

        # Transients
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        onset_count = len(onsets)

        return {
            "engine": "librosa",
            "bpm": round(tempo, 1),
            "key": detected_key,
            "scale": scale,
            "key_full": f"{detected_key} {scale}",
            "duration_seconds": round(duration, 1),
            "spectral_centroid_hz": round(spectral_centroid, 0),
            "spectral_bandwidth_hz": round(spectral_bandwidth, 0),
            "spectral_rolloff_hz": round(spectral_rolloff, 0),
            "avg_rms_energy": round(avg_rms, 4),
            "dynamic_range_db": round(dynamic_range_db, 1),
            "frequency_distribution": {"low_pct": low_pct, "mid_pct": mid_pct, "high_pct": high_pct},
            "zero_crossing_rate": round(zcr, 4),
            "onset_count": onset_count,
            "onsets_per_second": round(onset_count / duration, 2)
        }
    finally:
        os.unlink(tmp_path)


# ════════════════════════════════════════
# ENGINE 2: ESSENTIA — ML Music Analysis
# ════════════════════════════════════════
def analyze_with_essentia(audio_bytes: bytes) -> dict:
    """ML-based: genre, mood, instrument detection, danceability, vocal presence."""
    import essentia.standard as es
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        audio = es.MonoLoader(filename=tmp_path, sampleRate=44100)()
        result = {}

        try:
            rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
            result["bpm"] = round(float(bpm), 1)
            result["bpm_confidence"] = round(float(beats_confidence), 3)
        except:
            result["bpm"] = None
            result["bpm_confidence"] = None

        try:
            key_extractor = es.KeyExtractor()
            key, scale, key_strength = key_extractor(audio)
            result["key"] = key
            result["scale"] = scale
            result["key_full"] = f"{key} {scale}"
            result["key_strength"] = round(float(key_strength), 3)
        except:
            result["key"] = None
            result["scale"] = None
            result["key_full"] = None
            result["key_strength"] = None

        try:
            danceability_algo = es.Danceability()
            danceability, _ = danceability_algo(audio)
            result["danceability"] = round(float(danceability), 3)
        except:
            result["danceability"] = None

        try:
            energy = es.Energy()(audio)
            result["energy"] = round(float(energy), 4)
        except:
            result["energy"] = None

        try:
            loudness = es.Loudness()(audio)
            result["loudness"] = round(float(loudness), 4)
        except:
            result["loudness"] = None

        try:
            sc = es.SpectralComplexity()
            spectrum_algo = es.Spectrum()
            w = es.Windowing(type='hann')
            frame_gen = es.FrameGenerator(audio, frameSize=2048, hopSize=1024)
            complexities = []
            for frame in frame_gen:
                windowed = w(frame)
                spec = spectrum_algo(windowed)
                complexity = sc(spec)
                complexities.append(complexity)
            result["spectral_complexity"] = round(float(np.mean(complexities)), 2) if complexities else None
        except:
            result["spectral_complexity"] = None

        try:
            dissonance_algo = es.Dissonance()
            spectral_peaks = es.SpectralPeaks()
            frame_gen = es.FrameGenerator(audio, frameSize=2048, hopSize=1024)
            dissonances = []
            for frame in frame_gen:
                windowed = w(frame)
                spec = spectrum_algo(windowed)
                freqs, mags = spectral_peaks(spec)
                if len(freqs) > 1:
                    diss = dissonance_algo(freqs, mags)
                    dissonances.append(diss)
            result["dissonance"] = round(float(np.mean(dissonances)), 4) if dissonances else None
        except:
            result["dissonance"] = None

        try:
            dc = es.DynamicComplexity()
            dynamic_complexity, loudness_band = dc(audio)
            result["dynamic_complexity"] = round(float(dynamic_complexity), 2)
        except:
            result["dynamic_complexity"] = None

        result["engine"] = "essentia"
        return result
    except Exception as e:
        return {"engine": "essentia", "error": str(e)}
    finally:
        os.unlink(tmp_path)


# ════════════════════════════════════════
# ENGINE 3 (Pro): GEMINI PRO — Full Subjective Analysis
# ════════════════════════════════════════
GEMINI_ANALYSIS_PROMPT = """You are an expert music production analyst. Analyze this audio file.
Focus ONLY on qualities that require human-like listening judgment.
Do NOT estimate BPM, key, or any numerical measurements — those are handled by separate precision tools.

Respond ONLY in valid JSON format with NO other text:
{
  "genre": "detected genre(s) and sub-genres",
  "instruments": ["list", "of", "detected", "instruments/sound_sources"],
  "mood": "overall mood/atmosphere description",
  "structure": "song structure with approximate timestamps (intro, verse, chorus, drop, etc.)",
  "production_notes": "mixing quality, sound design, effects usage, overall polish",
  "vocal_type": "vocal characteristics if present, or 'instrumental'",
  "stereo_field": "stereo width, panning, spatial characteristics",
  "dynamics_description": "energy flow, tension/release patterns, build-ups and drops"
}"""

# ════════════════════════════════════════
# ENGINE 3 (Lite): GEMINI FLASH — Lightweight Analysis for Mode B
# ════════════════════════════════════════
GEMINI_FLASH_ANALYSIS_PROMPT = """You are an expert music analyst. Analyze this audio and identify its key musical characteristics.
Be concise but precise. Do NOT estimate BPM or key (handled separately).

Respond ONLY in valid JSON with NO other text:
{
  "genre": "primary genre and sub-genre (be specific, e.g. 'dark ambient doom' not just 'ambient')",
  "mood": "mood and atmosphere (2-4 words)",
  "instruments": ["main", "instruments", "detected"],
  "vocal_type": "vocal style or 'instrumental'",
  "production_style": "brief production character (e.g. 'lo-fi organic', 'polished EDM', 'raw garage')",
  "energy_level": "low/medium/high",
  "tempo_feel": "slow/moderate/fast (based on feel, not BPM)",
  "analysis_confidence": 1-10
}"""

# ════════════════════════════════════════
# REVERSE PROMPT PREDICTOR — Mode B Core
# ════════════════════════════════════════
REVERSE_PROMPT_TEMPLATE = """You are a Suno.ai prompt engineering expert.
Given the following audio analysis data, predict the most effective Suno prompt that would generate music matching these characteristics.

LIBROSA MEASUREMENTS:
{librosa_data}

ESSENTIA MEASUREMENTS:
{essentia_data}

GEMINI FLASH ANALYSIS:
{gemini_data}

TASK: Create a Suno-optimized prompt that would reproduce music with these characteristics.
Suno responds to: genre tags, mood descriptors, instrument mentions, BPM hints, key hints, production style words.
Avoid contradictory tokens (e.g., don't combine 'soft acoustic' with 'heavy distortion').

Respond ONLY in valid JSON with NO other text:
{{
  "predicted_prompt": "the complete Suno prompt string (50-150 chars)",
  "style_tags": ["individual", "style", "tokens", "used"],
  "bpm_token": "BPM hint included or null",
  "key_token": "key hint included or null",
  "confidence": 1-10,
  "reasoning": "brief explanation of key prompt decisions"
}}"""

GPT4O_ANALYSIS_PROMPT = """You are an expert music production analyst with deep knowledge of electronic music, hip-hop, jazz, rock, and world music sub-genres.
Analyze this audio file. Focus ONLY on qualities that require expert human-like listening judgment.
Do NOT estimate BPM, key, or any numerical measurements — those are handled by separate precision tools.

Respond ONLY in valid JSON format with NO other text:
{
  "genre": "detected genre(s) and sub-genres — be as specific as possible (e.g. 'dark electro house' not just 'electronic')",
  "instruments": ["list", "of", "every", "detected", "instrument", "and", "sound_source"],
  "mood": "overall mood/atmosphere — use production-specific vocabulary",
  "structure": "song structure with approximate timestamps",
  "production_notes": "mixing quality, distortion level, compression character, reverb type, sound design details",
  "vocal_type": "vocal characteristics if present, or 'instrumental'",
  "stereo_field": "stereo width, panning techniques, spatial effects",
  "dynamics_description": "energy flow, tension/release, transient character, sidechain behavior"
}"""


# ════════════════════════════════════════
# CLAUDE OPUS: Final Evaluation (Mode A)
# ════════════════════════════════════════
CLAUDE_EVALUATION_PROMPT = """You are evaluating how accurately an AI music generator (Suno) interpreted a style prompt.

ORIGINAL PROMPT given to Suno:
{prompt}

═══ ANALYSIS DATA FROM 4 INDEPENDENT ENGINES ═══
ENGINE 1 — LIBROSA (mathematically precise measurements):
{librosa_data}

ENGINE 2 — ESSENTIA (ML-based music classification):
{essentia_data}

ENGINE 3 — GEMINI 3.1 PRO (AI subjective listening analysis #1):
{gemini_data}

ENGINE 4 — GPT-4o AUDIO (AI subjective listening analysis #2):
{gpt4o_data}

═══ EVALUATION INSTRUCTIONS ═══
For BPM accuracy: Use LIBROSA's BPM as primary, cross-check with ESSENTIA's BPM.
For Key accuracy: Use LIBROSA's key as primary, cross-check with ESSENTIA's key. Note enharmonic equivalents (D#=Eb, etc).
For Genre accuracy: Cross-reference GEMINI and GPT-4o genre classifications. Where they agree, high confidence. Where they disagree, use ESSENTIA as tiebreaker.
For Mood accuracy: Compare ESSENTIA's danceability/energy/dissonance with BOTH Gemini and GPT-4o mood descriptions.
For Instrument accuracy: Cross-reference GEMINI and GPT-4o instrument lists. Instruments detected by both have high confidence.
For Structure accuracy: Cross-reference GEMINI and GPT-4o structure descriptions.
For Production quality: Cross-reference GEMINI and GPT-4o production notes.

═══ ENGINE RELIABILITY WEIGHTING ═══
IMPORTANT: Gemini 3.1 Pro tends to be MORE detailed and MORE accurate at genre/sub-genre classification and mood detection than GPT-4o Audio.
GPT-4o Audio tends to under-classify genres (e.g. calling "doom metal" just "post-rock") and use milder mood vocabulary.
When Gemini and GPT-4o disagree on genre or mood:
- Give Gemini's classification ~70% weight and GPT-4o ~30% weight
- Do NOT let GPT-4o's weaker genre/mood judgment drag down scores excessively

═══ PROMPT CONFLICT ANALYSIS ═══
CRITICAL: Before scoring, analyze the original prompt for INTERNAL CONTRADICTIONS between tokens.
Suno has a priority hierarchy:
1. Genre archetype energy > explicit BPM (e.g. "trailer music" implies 120+ BPM, overriding "85 BPM")
2. Energy tokens > mood tokens (e.g. "energetic" overrides "gentle")
3. Mainstream genre template > specific sub-genre request

If the prompt contains conflicting tokens, note this in your summary.
Include a "prompt_conflicts" field identifying any detected conflicts.

Respond ONLY in valid JSON:
{{
  "genre_accuracy": <1-10>,
  "bpm_accuracy": <1-10>,
  "key_accuracy": <1-10>,
  "instrument_accuracy": <1-10>,
  "mood_accuracy": <1-10>,
  "structure_accuracy": <1-10>,
  "overall_score": <1.0-10.0>,
  "engine_cross_check": {{
    "bpm_agreement": "librosa X vs essentia Y — agree/disagree",
    "key_agreement": "librosa X vs essentia Y — agree/disagree",
    "genre_agreement": "gemini 'X' vs gpt4o 'Y' — agree/disagree",
    "mood_agreement": "gemini 'X' vs gpt4o 'Y' — agree/disagree"
  }},
  "prompt_conflicts": [
    {{"conflict": "token A vs token B", "winner": "which token Suno prioritized", "reason": "why"}}
  ],
  "summary": "Brief explanation of what matched well and what didn't, including any prompt conflicts detected",
  "token_feedback": [
    {{"token": "specific word from prompt", "effectiveness": "high/medium/low", "reason": "why"}}
  ]
}}"""


# ════════════════════════════════════════
# CLAUDE OPUS: Re-evaluation (Mode B → Pro tier)
# ════════════════════════════════════════
CLAUDE_REEVAL_PROMPT = """You are evaluating how well a Suno.ai prompt would recreate a reference audio track.

PREDICTED SUNO PROMPT (reverse-engineered from the audio):
{predicted_prompt}

ORIGINAL AUDIO ANALYSIS:
ENGINE 1 — LIBROSA:
{librosa_data}

ENGINE 2 — ESSENTIA:
{essentia_data}

ENGINE 3 — GEMINI PRO (full analysis):
{gemini_pro_data}

═══ EVALUATION TASK ═══
Assess whether the predicted prompt accurately captures the musical essence of this audio.
If Suno were to generate music from this predicted prompt, how faithfully would it reproduce the original?

Respond ONLY in valid JSON:
{{
  "prompt_coverage_score": <1-10>,
  "genre_capture": <1-10>,
  "mood_capture": <1-10>,
  "instrument_capture": <1-10>,
  "missing_elements": ["elements", "not", "captured", "in", "prompt"],
  "over_specified": ["tokens", "that", "may", "confuse", "Suno"],
  "refined_prompt": "improved version of the predicted prompt",
  "fidelity_estimate": <1-10>,
  "notes": "key observations about prompt quality"
}}"""


# ════════════════════════════════════════
# API CALL HELPERS
# ════════════════════════════════════════
def extract_json(text: str) -> str:
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()


async def run_gemini(audio_bytes: bytes, mime_type: str = "audio/mpeg") -> str:
    """Full analysis with Gemini Pro."""
    model = genai.GenerativeModel(GEMINI_PRO_MODEL)
    response = model.generate_content([
        GEMINI_ANALYSIS_PROMPT,
        {"mime_type": mime_type, "data": audio_bytes}
    ])
    return response.text


async def run_gemini_flash(audio_bytes: bytes, mime_type: str = "audio/mpeg") -> str:
    """Lite analysis with Gemini Flash — cheaper, faster."""
    model = genai.GenerativeModel(GEMINI_FLASH_MODEL)
    response = model.generate_content([
        GEMINI_FLASH_ANALYSIS_PROMPT,
        {"mime_type": mime_type, "data": audio_bytes}
    ])
    return response.text


async def run_reverse_prompt(librosa_data: dict, essentia_data: dict, gemini_flash_data: str) -> dict:
    """Predict Suno prompt from audio analysis using Gemini Flash."""
    model = genai.GenerativeModel(GEMINI_FLASH_MODEL)
    prompt_text = REVERSE_PROMPT_TEMPLATE.format(
        librosa_data=json.dumps(librosa_data, indent=2),
        essentia_data=json.dumps(essentia_data, indent=2),
        gemini_data=gemini_flash_data
    )
    response = model.generate_content(prompt_text)
    raw = extract_json(response.text)
    try:
        return json.loads(raw)
    except:
        return {"predicted_prompt": raw, "confidence": 0, "error": "parse_failed"}


async def run_gemini_pro_full(audio_bytes: bytes, mime_type: str = "audio/mpeg") -> str:
    """Full Gemini Pro analysis (used in re-eval)."""
    return await run_gemini(audio_bytes, mime_type)


async def run_gpt4o(audio_bytes: bytes, mime_type: str = "audio/mpeg") -> str:
    """Send audio to GPT-4o Audio for subjective analysis."""
    client = OpenAI(api_key=OPENAI_KEY)
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    fmt = "mp3"
    if "wav" in mime_type:
        fmt = "wav"
    elif "mp4" in mime_type or "m4a" in mime_type:
        fmt = "mp4"
    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": GPT4O_ANALYSIS_PROMPT},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": fmt}}
            ]
        }],
        max_tokens=2000
    )
    return response.choices[0].message.content


async def run_claude(prompt: str, librosa_data: dict, essentia_data: dict, gemini_report: str, gpt4o_report: str) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2500,
        messages=[{
            "role": "user",
            "content": CLAUDE_EVALUATION_PROMPT.format(
                prompt=prompt,
                librosa_data=json.dumps(librosa_data, indent=2),
                essentia_data=json.dumps(essentia_data, indent=2),
                gemini_data=gemini_report,
                gpt4o_data=gpt4o_report
            )
        }]
    )
    text = message.content[0].text
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return json.loads(text.strip())


async def run_claude_reeval(predicted_prompt: str, librosa_data: dict, essentia_data: dict, gemini_pro_data: str) -> dict:
    """Claude Opus re-evaluation for Mode B high-quality tracks."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": CLAUDE_REEVAL_PROMPT.format(
                predicted_prompt=predicted_prompt,
                librosa_data=json.dumps(librosa_data, indent=2),
                essentia_data=json.dumps(essentia_data, indent=2),
                gemini_pro_data=gemini_pro_data
            )
        }]
    )
    text = message.content[0].text
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return json.loads(text.strip())


def compute_data_quality_score(librosa_data: dict, essentia_data: dict, gemini_flash: dict) -> float:
    """Heuristic quality score (0-10) to decide if re-eval is worthwhile."""
    score = 0.0

    # librosa quality
    if librosa_data.get("bpm") and librosa_data["bpm"] > 0:
        score += 1.5
    if librosa_data.get("key"):
        score += 1.0
    if librosa_data.get("duration_seconds", 0) > 30:
        score += 0.5

    # essentia quality
    if essentia_data.get("key_strength", 0) and essentia_data["key_strength"] > 0.5:
        score += 1.5
    if essentia_data.get("danceability") is not None:
        score += 0.5
    if not essentia_data.get("error"):
        score += 0.5

    # gemini flash quality
    if isinstance(gemini_flash, dict):
        if gemini_flash.get("genre") and gemini_flash["genre"].lower() not in ["unknown", "error", ""]:
            score += 1.5
        if gemini_flash.get("analysis_confidence", 0) >= 7:
            score += 1.5
        if gemini_flash.get("instruments") and len(gemini_flash["instruments"]) > 1:
            score += 0.5

    return round(min(score, 10.0), 1)


# ════════════════════════════════════════
# SUPABASE HELPERS
# ════════════════════════════════════════
async def save_to_supabase(data: dict):
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/audio_analysis",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation"
            },
            json=data
        )
        return r.json() if r.status_code < 300 else None


async def fetch_from_supabase(analysis_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/audio_analysis?id=eq.{analysis_id}&select=*",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
        )
        data = r.json()
        return data[0] if data else None


async def update_supabase(analysis_id: int, updates: dict):
    async with httpx.AsyncClient() as client:
        r = await client.patch(
            f"{SUPABASE_URL}/rest/v1/audio_analysis?id=eq.{analysis_id}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json"
            },
            json=updates
        )
        return r.status_code < 300


# ════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "suno-audio-analyzer",
        "version": "5.0-mode-b",
        "modes": {
            "mode_a": "Prompt + Audio → Quad Engine → Claude Opus evaluation",
            "mode_b": "Audio only → librosa + Essentia + Gemini Flash → Reverse prompt prediction"
        },
        "engines": {
            "lite": ["librosa", "essentia", GEMINI_FLASH_MODEL],
            "pro": ["librosa", "essentia", GEMINI_PRO_MODEL, "gpt-4o-audio", "claude-opus-4-6"]
        },
        "active_analyses": MAX_CONCURRENT - analysis_semaphore._value,
        "max_concurrent": MAX_CONCURRENT
    }


# ────────────────────────────────────────
# MODE A: Original — Prompt + Audio → Full evaluation
# ────────────────────────────────────────
@app.post("/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    prompt_type: str = Form("style"),
    prompt_id: str = Form(None),
    ip: str = Form("unknown")
):
    """Mode A: Quad-engine analysis with prompt → Claude Opus evaluation."""
    if not file.content_type or "audio" not in file.content_type:
        raise HTTPException(400, "File must be an audio file (mp3, wav, etc.)")

    audio_bytes = await file.read()
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum 25MB.")

    try:
        async with analysis_semaphore:
            try:
                librosa_result = analyze_with_librosa(audio_bytes)
            except Exception as e:
                librosa_result = {"engine": "librosa", "error": str(e)}

            try:
                essentia_result = analyze_with_essentia(audio_bytes)
            except Exception as e:
                essentia_result = {"engine": "essentia", "error": str(e)}

            try:
                gemini_raw = await run_gemini(audio_bytes, file.content_type)
                gemini_report = extract_json(gemini_raw)
            except Exception as e:
                gemini_report = json.dumps({"engine": "gemini", "error": str(e)})

            try:
                gpt4o_raw = await run_gpt4o(audio_bytes, file.content_type)
                gpt4o_report = extract_json(gpt4o_raw)
            except Exception as e:
                gpt4o_report = json.dumps({"engine": "gpt4o", "error": str(e)})

            try:
                claude_eval = await run_claude(prompt, librosa_result, essentia_result, gemini_report, gpt4o_report)
            except Exception as e:
                claude_eval = {
                    "genre_accuracy": None, "bpm_accuracy": None, "key_accuracy": None,
                    "instrument_accuracy": None, "mood_accuracy": None, "structure_accuracy": None,
                    "overall_score": None, "engine_cross_check": {}, "prompt_conflicts": [],
                    "summary": f"Claude evaluation failed: {str(e)}", "token_feedback": []
                }

            full_report = json.dumps({
                "librosa": librosa_result,
                "essentia": essentia_result,
                "gemini": json.loads(gemini_report) if gemini_report.startswith("{") else gemini_report,
                "gpt4o": json.loads(gpt4o_report) if gpt4o_report.startswith("{") else gpt4o_report
            })

            db_result = {
                "ip": ip,
                "prompt_id": int(prompt_id) if prompt_id and prompt_id.isdigit() else None,
                "original_prompt": prompt,
                "gemini_report": full_report,
                "genre_accuracy": claude_eval.get("genre_accuracy"),
                "bpm_accuracy": claude_eval.get("bpm_accuracy"),
                "instrument_accuracy": claude_eval.get("instrument_accuracy"),
                "mood_accuracy": claude_eval.get("mood_accuracy"),
                "structure_accuracy": claude_eval.get("structure_accuracy"),
                "overall_score": claude_eval.get("overall_score"),
                "prompt_type": prompt_type,
                "analysis_mode": "prompt_eval"
            }
            await save_to_supabase(db_result)

            return {
                "librosa": librosa_result,
                "essentia": essentia_result,
                "gemini": json.loads(gemini_report) if gemini_report.startswith("{") else gemini_report,
                "gpt4o": json.loads(gpt4o_report) if gpt4o_report.startswith("{") else gpt4o_report,
                "evaluation": claude_eval,
                "saved": True
            }

    except asyncio.TimeoutError:
        raise HTTPException(503, "Server busy. Please try again in a moment.")


# ────────────────────────────────────────
# MODE B: Audio only → reverse prompt prediction
# ────────────────────────────────────────
@app.post("/analyze-audio")
async def analyze_audio_only(
    file: UploadFile = File(...),
    ip: str = Form("unknown"),
    label: str = Form("")  # optional user label (e.g., "my_stem_01")
):
    """
    Mode B: Audio-only analysis without a prompt.
    Uses librosa + Essentia + Gemini Flash (lite tier).
    Outputs predicted Suno prompt + data quality score.
    High-quality results (score >= threshold) are flagged for re-eval.
    """
    if not file.content_type or "audio" not in file.content_type:
        raise HTTPException(400, "File must be an audio file (mp3, wav, etc.)")

    audio_bytes = await file.read()
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum 25MB.")

    try:
        async with analysis_semaphore:
            # Engine 1: librosa
            try:
                librosa_result = analyze_with_librosa(audio_bytes)
            except Exception as e:
                librosa_result = {"engine": "librosa", "error": str(e)}

            # Engine 2: Essentia
            try:
                essentia_result = analyze_with_essentia(audio_bytes)
            except Exception as e:
                essentia_result = {"engine": "essentia", "error": str(e)}

            # Engine 3 (Lite): Gemini Flash subjective analysis
            try:
                flash_raw = await run_gemini_flash(audio_bytes, file.content_type)
                flash_report_str = extract_json(flash_raw)
                flash_report = json.loads(flash_report_str) if flash_report_str.startswith("{") else {"raw": flash_report_str}
            except Exception as e:
                flash_report = {"error": str(e)}
                flash_report_str = json.dumps(flash_report)

            # Reverse prompt prediction
            try:
                predicted = await run_reverse_prompt(librosa_result, essentia_result, flash_report_str)
            except Exception as e:
                predicted = {"predicted_prompt": "", "confidence": 0, "error": str(e)}

            # Data quality score
            quality_score = compute_data_quality_score(librosa_result, essentia_result, flash_report)
            reeval_recommended = quality_score >= REEVAL_SCORE_THRESHOLD

            # Save to Supabase
            db_result = {
                "ip": ip,
                "original_prompt": predicted.get("predicted_prompt", ""),
                "gemini_report": json.dumps({
                    "librosa": librosa_result,
                    "essentia": essentia_result,
                    "gemini_flash": flash_report,
                    "predicted": predicted
                }),
                "analysis_mode": "audio_only",
                "prompt_type": "predicted",
                "overall_score": None,  # not evaluated yet
            }
            saved = await save_to_supabase(db_result)
            saved_id = saved[0].get("id") if saved and isinstance(saved, list) else None

            return {
                "mode": "audio_only",
                "librosa": librosa_result,
                "essentia": essentia_result,
                "gemini_flash": flash_report,
                "predicted_prompt": predicted,
                "quality_score": quality_score,
                "reeval_recommended": reeval_recommended,
                "analysis_id": saved_id,
                "label": label
            }

    except asyncio.TimeoutError:
        raise HTTPException(503, "Server busy. Please try again in a moment.")


# ────────────────────────────────────────
# ADMIN: Re-evaluation of Mode B results
# (Pro tier — Gemini Pro + Claude Opus, internal use only)
# ────────────────────────────────────────
@app.post("/admin/reeval")
async def admin_reeval(
    analysis_id: int = Form(...),
    x_admin_key: str = Header(None)
):
    """
    Admin-only: Re-evaluate a Mode B audio analysis using Gemini Pro + Claude Opus.
    Requires X-Admin-Key header matching ADMIN_KEY env var.
    Updates the Supabase record with detailed re-eval results.
    """
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "Unauthorized")

    record = await fetch_from_supabase(analysis_id)
    if not record:
        raise HTTPException(404, f"Analysis {analysis_id} not found")

    if record.get("analysis_mode") != "audio_only":
        raise HTTPException(400, "Re-eval is only for audio_only analyses")

    raw_report = json.loads(record.get("gemini_report", "{}"))
    librosa_data = raw_report.get("librosa", {})
    essentia_data = raw_report.get("essentia", {})
    predicted = raw_report.get("predicted", {})
    predicted_prompt = predicted.get("predicted_prompt", "")

    if not predicted_prompt:
        raise HTTPException(400, "No predicted prompt found in this record")

    # NOTE: We don't have the original audio bytes stored — only the analysis.
    # So re-eval uses the stored analysis data + Gemini Flash report as proxy.
    # For a higher-fidelity re-eval, store audio bytes in Supabase Storage (future improvement).
    flash_data = raw_report.get("gemini_flash", {})

    # Run Gemini Pro on stored analysis context (text-only, since we don't have audio)
    # This is a text-based re-analysis using Gemini Pro
    try:
        model = genai.GenerativeModel(GEMINI_PRO_MODEL)
        gemini_pro_prompt = f"""Based on this audio analysis data, provide a detailed musical assessment:

LIBROSA: {json.dumps(librosa_data, indent=2)}
ESSENTIA: {json.dumps(essentia_data, indent=2)}
GEMINI FLASH ANALYSIS: {json.dumps(flash_data, indent=2)}

Respond in valid JSON:
{{
  "genre": "detailed genre classification",
  "mood": "detailed mood analysis",
  "instruments": ["full", "instrument", "list"],
  "production_notes": "production quality assessment",
  "vocal_type": "vocal analysis or 'instrumental'",
  "structure_notes": "inferred song structure",
  "suno_generation_notes": "how Suno would likely interpret these characteristics"
}}"""
        gemini_pro_response = model.generate_content(gemini_pro_prompt)
        gemini_pro_data = extract_json(gemini_pro_response.text)
    except Exception as e:
        gemini_pro_data = json.dumps({"error": str(e)})

    # Claude Opus re-evaluation
    try:
        claude_reeval = await run_claude_reeval(predicted_prompt, librosa_data, essentia_data, gemini_pro_data)
    except Exception as e:
        claude_reeval = {"error": str(e), "fidelity_estimate": None}

    reeval_data = {
        "gemini_pro_reanalysis": json.loads(gemini_pro_data) if gemini_pro_data.startswith("{") else gemini_pro_data,
        "claude_reeval": claude_reeval,
        "reeval_timestamp": asyncio.get_event_loop().time()
    }

    # Update Supabase record
    await update_supabase(analysis_id, {
        "overall_score": claude_reeval.get("fidelity_estimate"),
        "gemini_report": json.dumps({**raw_report, "reeval": reeval_data})
    })

    return {
        "analysis_id": analysis_id,
        "predicted_prompt": predicted_prompt,
        "gemini_pro_reanalysis": json.loads(gemini_pro_data) if gemini_pro_data.startswith("{") else gemini_pro_data,
        "claude_reeval": claude_reeval,
        "fidelity_estimate": claude_reeval.get("fidelity_estimate")
    }


@app.get("/admin/reeval-queue")
async def get_reeval_queue(
    x_admin_key: str = Header(None),
    limit: int = 20
):
    """Admin: List Mode B analyses flagged for re-evaluation (quality_score >= threshold)."""
    if not ADMIN_KEY or x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "Unauthorized")

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/audio_analysis"
            f"?analysis_mode=eq.audio_only&overall_score=is.null&select=id,created_at,original_prompt,gemini_report"
            f"&order=created_at.desc&limit={limit}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
        )
        records = r.json()

    # Filter by quality score stored in gemini_report JSON
    flagged = []
    for rec in records:
        try:
            report = json.loads(rec.get("gemini_report", "{}"))
            flash = report.get("gemini_flash", {})
            librosa = report.get("librosa", {})
            essentia = report.get("essentia", {})
            score = compute_data_quality_score(librosa, essentia, flash)
            if score >= REEVAL_SCORE_THRESHOLD:
                flagged.append({
                    "id": rec["id"],
                    "created_at": rec.get("created_at"),
                    "predicted_prompt": rec.get("original_prompt"),
                    "quality_score": score
                })
        except:
            pass

    return {
        "reeval_threshold": REEVAL_SCORE_THRESHOLD,
        "flagged_count": len(flagged),
        "records": flagged
    }
