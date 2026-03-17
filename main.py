from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import anthropic
import httpx
import json
import os
import tempfile
import re
import numpy as np

app = FastAPI(title="Suno Audio Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config from environment
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jgfvwfalxnrdujaoqoiq.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Gemini
genai.configure(api_key=GEMINI_KEY)

# ─── LIBROSA: Objective Audio Analysis ───

def analyze_audio_objective(audio_bytes: bytes) -> dict:
    """Use librosa for mathematically accurate BPM, key, frequency analysis."""
    import librosa
    import io

    # Load audio from bytes
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # BPM detection
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)

        # Key detection via chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key_idx = int(np.argmax(chroma_mean))
        detected_key = key_names[detected_key_idx]

        # Major/Minor detection
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        # Rotate profiles to match detected key
        major_corr = float(np.corrcoef(chroma_mean, np.roll(major_profile, detected_key_idx))[0, 1])
        minor_corr = float(np.corrcoef(chroma_mean, np.roll(minor_profile, detected_key_idx))[0, 1])
        scale = "major" if major_corr > minor_corr else "minor"

        # Spectral analysis
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        avg_rms = float(np.mean(rms))
        max_rms = float(np.max(rms))
        dynamic_range_db = float(20 * np.log10(max_rms / (avg_rms + 1e-10)))

        # Frequency band energy distribution
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        low_mask = freqs < 250
        mid_mask = (freqs >= 250) & (freqs < 4000)
        high_mask = freqs >= 4000
        total_energy = float(np.sum(S ** 2)) + 1e-10
        low_pct = round(float(np.sum(S[low_mask] ** 2)) / total_energy * 100, 1)
        mid_pct = round(float(np.sum(S[mid_mask] ** 2)) / total_energy * 100, 1)
        high_pct = round(float(np.sum(S[high_mask] ** 2)) / total_energy * 100, 1)

        # Zero crossing rate (indicates percussiveness)
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        # Onset detection (number of transients/hits)
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        onset_count = len(onsets)

        return {
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
            "frequency_distribution": {
                "low_pct": low_pct,
                "mid_pct": mid_pct,
                "high_pct": high_pct
            },
            "zero_crossing_rate": round(zcr, 4),
            "onset_count": onset_count,
            "onsets_per_second": round(onset_count / duration, 2)
        }
    finally:
        os.unlink(tmp_path)


# ─── GEMINI: Subjective Audio Analysis ───

GEMINI_ANALYSIS_PROMPT = """You are an expert music production analyst. Analyze this audio file.
Focus ONLY on subjective qualities that require human-like judgment.
Do NOT estimate BPM or key — those will be measured separately with precise tools.

Respond ONLY in valid JSON format with NO other text:
{
  "genre": "detected genre(s) and sub-genres",
  "instruments": ["list", "of", "detected", "instruments"],
  "mood": "overall mood/atmosphere description",
  "structure": "song structure description (intro, verse, chorus, drop, etc.) with approximate timestamps",
  "production_notes": "production quality, mixing characteristics, sound design details",
  "vocal_type": "vocal characteristics if present, or 'instrumental'",
  "stereo_field": "stereo width and spatial characteristics",
  "dynamics_description": "energy flow and tension/release patterns"
}"""


CLAUDE_EVALUATION_PROMPT = """You are evaluating how accurately an AI music generator (Suno) interpreted a style prompt.

ORIGINAL PROMPT given to Suno:
{prompt}

OBJECTIVE AUDIO MEASUREMENTS (mathematically precise):
{objective}

SUBJECTIVE AUDIO ANALYSIS by Gemini AI:
{subjective}

Rate the accuracy of the generated music compared to the original prompt intent on each dimension (1-10 scale).
Use the OBJECTIVE measurements for BPM and key accuracy — these are mathematically measured, not estimated.
Use the SUBJECTIVE analysis for genre, mood, instruments, and structure accuracy.

Respond ONLY in valid JSON:
{{
  "genre_accuracy": <1-10>,
  "bpm_accuracy": <1-10>,
  "key_accuracy": <1-10>,
  "instrument_accuracy": <1-10>,
  "mood_accuracy": <1-10>,
  "structure_accuracy": <1-10>,
  "overall_score": <1.0-10.0>,
  "summary": "Brief explanation of what matched well and what didn't",
  "token_feedback": [
    {{"token": "specific word from prompt", "effectiveness": "high/medium/low", "reason": "why"}}
  ]
}}"""


async def save_to_supabase(data: dict):
    """Save analysis result to Supabase."""
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


async def analyze_with_gemini(audio_bytes: bytes, mime_type: str = "audio/mpeg") -> str:
    """Send audio to Gemini for subjective analysis only."""
    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content([
        GEMINI_ANALYSIS_PROMPT,
        {"mime_type": mime_type, "data": audio_bytes}
    ])
    return response.text


async def evaluate_with_claude(original_prompt: str, objective_data: dict, subjective_report: str) -> dict:
    """Send objective + subjective analysis + original prompt to Claude for evaluation."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": CLAUDE_EVALUATION_PROMPT.format(
                prompt=original_prompt,
                objective=json.dumps(objective_data, indent=2),
                subjective=subjective_report
            )
        }]
    )

    text = message.content[0].text
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return json.loads(text.strip())


def extract_json(text: str) -> str:
    """Extract JSON from potentially messy AI response."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "suno-audio-analyzer", "version": "2.0-librosa"}


@app.post("/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    prompt_type: str = Form("style"),
    prompt_id: int = Form(None),
    ip: str = Form("unknown")
):
    """Analyze uploaded mp3 file against original prompt."""
    if not file.content_type or "audio" not in file.content_type:
        raise HTTPException(400, "File must be an audio file (mp3, wav, etc.)")

    audio_bytes = await file.read()
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum 25MB.")

    # Step 1: Librosa - objective measurements
    try:
        objective = analyze_audio_objective(audio_bytes)
    except Exception as e:
        raise HTTPException(500, f"Audio analysis failed: {str(e)}")

    # Step 2: Gemini - subjective analysis
    try:
        gemini_raw = await analyze_with_gemini(audio_bytes, file.content_type)
        gemini_report = extract_json(gemini_raw)
    except Exception as e:
        gemini_report = json.dumps({"error": f"Gemini analysis failed: {str(e)}"})

    # Step 3: Claude - evaluate prompt vs result
    try:
        claude_eval = await evaluate_with_claude(prompt, objective, gemini_report)
    except Exception as e:
        claude_eval = {
            "genre_accuracy": None, "bpm_accuracy": None, "key_accuracy": None,
            "instrument_accuracy": None, "mood_accuracy": None,
            "structure_accuracy": None, "overall_score": None,
            "summary": f"Claude evaluation failed: {str(e)}",
            "token_feedback": []
        }

    # Step 4: Save to Supabase
    full_report = json.dumps({
        "objective": objective,
        "subjective": json.loads(gemini_report) if gemini_report.startswith("{") else gemini_report
    })

    result = {
        "ip": ip,
        "prompt_id": prompt_id,
        "original_prompt": prompt,
        "gemini_report": full_report,
        "genre_accuracy": claude_eval.get("genre_accuracy"),
        "bpm_accuracy": claude_eval.get("bpm_accuracy"),
        "instrument_accuracy": claude_eval.get("instrument_accuracy"),
        "mood_accuracy": claude_eval.get("mood_accuracy"),
        "structure_accuracy": claude_eval.get("structure_accuracy"),
        "overall_score": claude_eval.get("overall_score"),
        "prompt_type": prompt_type
    }

    await save_to_supabase(result)

    return {
        "objective_analysis": objective,
        "subjective_analysis": json.loads(gemini_report) if gemini_report.startswith("{") else gemini_report,
        "evaluation": claude_eval,
        "saved": True
    }


@app.post("/analyze-url")
async def analyze_url(
    url: str = Form(...),
    prompt: str = Form(...),
    prompt_type: str = Form("style"),
    prompt_id: int = Form(None),
    ip: str = Form("unknown")
):
    """Analyze Suno song by URL."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            song_id = url.rstrip("/").split("/")[-1]
            audio_url = f"https://cdn1.suno.ai/{song_id}.mp3"
            r = await client.get(audio_url)
            if r.status_code != 200:
                audio_url = f"https://cdn2.suno.ai/{song_id}.mp3"
                r = await client.get(audio_url)
            if r.status_code != 200:
                raise HTTPException(400, "Could not download audio from Suno URL. Please upload mp3 directly.")
            audio_bytes = r.content
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch audio: {str(e)}. Please upload mp3 directly.")

    # Same pipeline
    try:
        objective = analyze_audio_objective(audio_bytes)
    except Exception as e:
        raise HTTPException(500, f"Audio analysis failed: {str(e)}")

    try:
        gemini_raw = await analyze_with_gemini(audio_bytes)
        gemini_report = extract_json(gemini_raw)
    except Exception as e:
        gemini_report = json.dumps({"error": f"Gemini analysis failed: {str(e)}"})

    try:
        claude_eval = await evaluate_with_claude(prompt, objective, gemini_report)
    except Exception as e:
        claude_eval = {
            "genre_accuracy": None, "bpm_accuracy": None, "key_accuracy": None,
            "instrument_accuracy": None, "mood_accuracy": None,
            "structure_accuracy": None, "overall_score": None,
            "summary": f"Claude evaluation failed: {str(e)}",
            "token_feedback": []
        }

    full_report = json.dumps({
        "objective": objective,
        "subjective": json.loads(gemini_report) if gemini_report.startswith("{") else gemini_report
    })

    result = {
        "ip": ip, "prompt_id": prompt_id,
        "original_prompt": prompt, "gemini_report": full_report,
        "genre_accuracy": claude_eval.get("genre_accuracy"),
        "bpm_accuracy": claude_eval.get("bpm_accuracy"),
        "instrument_accuracy": claude_eval.get("instrument_accuracy"),
        "mood_accuracy": claude_eval.get("mood_accuracy"),
        "structure_accuracy": claude_eval.get("structure_accuracy"),
        "overall_score": claude_eval.get("overall_score"),
        "suno_url": url, "prompt_type": prompt_type
    }

    await save_to_supabase(result)

    return {
        "objective_analysis": objective,
        "subjective_analysis": json.loads(gemini_report) if gemini_report.startswith("{") else gemini_report,
        "evaluation": claude_eval,
        "saved": True
    }
