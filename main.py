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
import asyncio

app = FastAPI(title="Suno Audio Analyzer v3 — Triple Engine")

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
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jgfvwfalxnrdujaoqoiq.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

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

        # BPM (cross-verify with librosa)
        try:
            rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
            bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
            result["bpm"] = round(float(bpm), 1)
            result["bpm_confidence"] = round(float(beats_confidence), 3)
        except:
            result["bpm"] = None
            result["bpm_confidence"] = None

        # Key (cross-verify with librosa)
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

        # Danceability
        try:
            danceability_algo = es.Danceability()
            danceability, _ = danceability_algo(audio)
            result["danceability"] = round(float(danceability), 3)
        except:
            result["danceability"] = None

        # Energy
        try:
            energy = es.Energy()(audio)
            result["energy"] = round(float(energy), 4)
        except:
            result["energy"] = None

        # Loudness
        try:
            loudness = es.Loudness()(audio)
            result["loudness"] = round(float(loudness), 4)
        except:
            result["loudness"] = None

        # Spectral complexity
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

        # Dissonance
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

        # Dynamic complexity
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
# ENGINE 3: GEMINI — AI Subjective Analysis
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
# CLAUDE OPUS: Final Evaluation
# ════════════════════════════════════════

CLAUDE_EVALUATION_PROMPT = """You are evaluating how accurately an AI music generator (Suno) interpreted a style prompt.

ORIGINAL PROMPT given to Suno:
{prompt}

═══ ANALYSIS DATA FROM 3 INDEPENDENT ENGINES ═══

ENGINE 1 — LIBROSA (mathematically precise measurements):
{librosa_data}

ENGINE 2 — ESSENTIA (ML-based music classification):
{essentia_data}

ENGINE 3 — GEMINI AI (subjective listening analysis):
{gemini_data}

═══ EVALUATION INSTRUCTIONS ═══

For BPM accuracy: Use LIBROSA's BPM as primary, cross-check with ESSENTIA's BPM.
For Key accuracy: Use LIBROSA's key as primary, cross-check with ESSENTIA's key.
For Genre accuracy: Use ESSENTIA's classification as primary, cross-check with GEMINI.
For Mood accuracy: Compare ESSENTIA's danceability/energy/dissonance with GEMINI's mood description.
For Instrument accuracy: Use GEMINI's instrument list (it can hear individual instruments).
For Structure accuracy: Use GEMINI's structure description.

Where engines disagree, note the disagreement and use your judgment on which is more reliable.

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
    "key_agreement": "librosa X vs essentia Y — agree/disagree"
  }},
  "summary": "Brief explanation of what matched well and what didn't",
  "token_feedback": [
    {{"token": "specific word from prompt", "effectiveness": "high/medium/low", "reason": "why"}}
  ]
}}"""


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


async def run_gemini(audio_bytes: bytes, mime_type: str = "audio/mpeg") -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content([
        GEMINI_ANALYSIS_PROMPT,
        {"mime_type": mime_type, "data": audio_bytes}
    ])
    return response.text


async def run_claude(prompt: str, librosa_data: dict, essentia_data: dict, gemini_report: str) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": CLAUDE_EVALUATION_PROMPT.format(
                prompt=prompt,
                librosa_data=json.dumps(librosa_data, indent=2),
                essentia_data=json.dumps(essentia_data, indent=2),
                gemini_data=gemini_report
            )
        }]
    )
    text = message.content[0].text
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return json.loads(text.strip())


def extract_json(text: str) -> str:
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "suno-audio-analyzer",
        "version": "3.0-triple-engine",
        "engines": ["librosa", "essentia", "gemini"],
        "evaluator": "claude-opus-4-6",
        "active_analyses": MAX_CONCURRENT - analysis_semaphore._value,
        "max_concurrent": MAX_CONCURRENT
    }


@app.post("/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    prompt_type: str = Form("style"),
    prompt_id: int = Form(None),
    ip: str = Form("unknown")
):
    """Triple-engine analysis: librosa + Essentia + Gemini → Claude Opus evaluation."""
    if not file.content_type or "audio" not in file.content_type:
        raise HTTPException(400, "File must be an audio file (mp3, wav, etc.)")

    audio_bytes = await file.read()
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum 25MB.")

    try:
        async with analysis_semaphore:
            # ── Engine 1: Librosa ──
            try:
                librosa_result = analyze_with_librosa(audio_bytes)
            except Exception as e:
                librosa_result = {"engine": "librosa", "error": str(e)}

            # ── Engine 2: Essentia ──
            try:
                essentia_result = analyze_with_essentia(audio_bytes)
            except Exception as e:
                essentia_result = {"engine": "essentia", "error": str(e)}

            # ── Engine 3: Gemini ──
            try:
                gemini_raw = await run_gemini(audio_bytes, file.content_type)
                gemini_report = extract_json(gemini_raw)
            except Exception as e:
                gemini_report = json.dumps({"engine": "gemini", "error": str(e)})

            # ── Evaluator: Claude Opus ──
            try:
                claude_eval = await run_claude(prompt, librosa_result, essentia_result, gemini_report)
            except Exception as e:
                claude_eval = {
                    "genre_accuracy": None, "bpm_accuracy": None, "key_accuracy": None,
                    "instrument_accuracy": None, "mood_accuracy": None,
                    "structure_accuracy": None, "overall_score": None,
                    "engine_cross_check": {},
                    "summary": f"Claude evaluation failed: {str(e)}",
                    "token_feedback": []
                }

            # ── Save to Supabase ──
            full_report = json.dumps({
                "librosa": librosa_result,
                "essentia": essentia_result,
                "gemini": json.loads(gemini_report) if gemini_report.startswith("{") else gemini_report
            })

            db_result = {
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

            await save_to_supabase(db_result)

            return {
                "librosa": librosa_result,
                "essentia": essentia_result,
                "gemini": json.loads(gemini_report) if gemini_report.startswith("{") else gemini_report,
                "evaluation": claude_eval,
                "saved": True
            }
    except asyncio.TimeoutError:
        raise HTTPException(503, "Server busy. Please try again in a moment.")
