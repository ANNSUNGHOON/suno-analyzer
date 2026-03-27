from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from pydantic import BaseModel
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

app = FastAPI(title="Suno Audio Analyzer v7.2 — Quad Engine + RAG")

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

        # BPM — with half-time correction
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)
        # librosa often doubles BPM for slow/mid-tempo tracks (e.g. 75 BPM → 150 BPM)
        # Heuristic: if tempo > 140 and half-tempo is in common range (60-120), prefer half
        tempo_raw = tempo
        tempo_half = tempo / 2
        if tempo > 140 and 55 <= tempo_half <= 130:
            tempo = tempo_half
        # Store both for analysis
        tempo_corrected = tempo

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
            "bpm": round(tempo_corrected, 1),
            "bpm_raw": round(tempo_raw, 1),
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
# ENGINE: CLAP — Prompt↔Audio Direct Alignment
# ════════════════════════════════════════
_clap_model = None

def _load_clap():
    """Lazy-load LAION-CLAP model (first call downloads ~600MB checkpoint)."""
    global _clap_model
    if _clap_model is None:
        import laion_clap
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        _clap_model.load_ckpt()
        print("[INFO] CLAP model loaded successfully")
    return _clap_model


def get_clap_alignment(audio_bytes: bytes, text_prompt: str) -> dict:
    """CLAP: Compute text↔audio cosine similarity in shared embedding space."""
    import torch
    import librosa as lr

    try:
        model = _load_clap()
    except Exception as e:
        return {"engine": "clap", "error": f"Model load failed: {e}"}

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        # CLAP needs 48kHz mono
        y, _ = lr.load(tmp_path, sr=48000, mono=True)
        audio_data = y.reshape(1, -1)

        with torch.no_grad():
            audio_embed = model.get_audio_embedding_from_data(audio_data, use_tensor=True)
            text_embed = model.get_text_embedding([text_prompt], use_tensor=True)

        sim = float(torch.nn.functional.cosine_similarity(audio_embed, text_embed, dim=1)[0])

        return {
            "engine": "clap",
            "cosine_similarity": round(sim, 4),
            "alignment_score_10": round(max(0.0, min(1.0, sim)) * 10, 1),
        }
    except Exception as e:
        return {"engine": "clap", "error": str(e)}
    finally:
        os.unlink(tmp_path)


# ════════════════════════════════════════
# CLAP MUSIC TAGGER — Zero-shot MTT vocabulary (MusiCNN functional equivalent)
# ════════════════════════════════════════
# MagnaTagATune 50-tag vocabulary (same as MusiCNN MTT model)
MTT_TAGS = [
    "guitar", "classical", "slow", "techno", "strings", "drums", "electronic",
    "rock", "fast", "piano", "ambient", "beat", "violin", "vocal", "synth",
    "female vocal", "indian", "opera", "male vocal", "singing", "no vocals",
    "harpsichord", "loud", "quiet", "flute", "choir", "jazz", "metal", "country",
    "dance", "new age", "hip hop", "smooth", "cello", "orchestral", "heavy",
    "reggae", "light", "funk", "folk", "bass", "trumpet", "saxophone",
    "keyboard", "acoustic", "pop", "soft", "energetic", "dark", "upbeat",
]


def score_clap_music_tagger(audio_bytes: bytes, prompt: str) -> dict:
    """Zero-shot music tagging via CLAP + MTT vocabulary. MusiCNN functional equivalent.
    Audio embed once → cosine sim against 50 fixed music tags → top tags → prompt match score.
    """
    import torch
    import torch.nn.functional as F
    import librosa as lr

    try:
        model = _load_clap()
    except Exception as e:
        return {"engine": "clap_tagger", "error": f"Model load failed: {e}"}

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, _ = lr.load(tmp_path, sr=48000, mono=True)
        audio_data = y.reshape(1, -1)

        with torch.no_grad():
            audio_embed = model.get_audio_embedding_from_data(audio_data, use_tensor=True)  # [1, 512]
            text_embeds = model.get_text_embedding(MTT_TAGS, use_tensor=True)               # [N, 512]
            sims = F.cosine_similarity(audio_embed.expand(len(MTT_TAGS), -1), text_embeds, dim=1)
            sims_np = sims.cpu().numpy()

        # Top tags above threshold
        top_indices = sims_np.argsort()[::-1][:10]
        detected = [(MTT_TAGS[i], round(float(sims_np[i]), 3)) for i in top_indices if sims_np[i] > 0.15]
        detected_names = set(tag for tag, _ in detected)

        # Prompt match score
        prompt_tokens = [w.strip().lower() for w in re.split(r'[,\s]+', prompt) if len(w.strip()) > 2]
        if not prompt_tokens:
            tag_match_score = 0.0
        else:
            matches = sum(1 for tok in prompt_tokens
                          if any(tok in tag or tag in tok for tag in detected_names))
            tag_match_score = round(min(10.0, (matches / len(prompt_tokens)) * 15), 1)

        return {
            "engine": "clap_tagger",
            "top_tags": detected[:5],
            "tag_match_score": tag_match_score,
        }
    except Exception as e:
        return {"engine": "clap_tagger", "error": str(e)}
    finally:
        os.unlink(tmp_path)


# ════════════════════════════════════════
# GEMINI FLASH TAG MATCH — Music tag keyword scoring
# ════════════════════════════════════════
def score_gemini_tag_match(prompt: str, flash_report: dict) -> float:
    """Score how well Gemini Flash's music tags match the original prompt keywords."""
    if not flash_report or flash_report.get("error"):
        return 0.0
    # Collect all detected tags: genre, mood, instruments, vocal, production
    detected_tokens = set()
    for field in ["genre", "mood", "vocal_type", "production_style", "energy_level", "tempo_feel"]:
        val = flash_report.get(field, "") or ""
        for tok in re.split(r'[,\s/\-]+', val.lower()):
            if len(tok) > 2:
                detected_tokens.add(tok)
    instruments = flash_report.get("instruments", []) or []
    for inst in instruments:
        for tok in re.split(r'[,\s]+', inst.lower()):
            if len(tok) > 2:
                detected_tokens.add(tok)

    # Prompt keywords
    prompt_tokens = [w.strip().lower() for w in re.split(r'[,\s]+', prompt) if len(w.strip()) > 3]
    if not prompt_tokens:
        return 0.0
    matches = sum(1 for tok in prompt_tokens if any(tok in dt or dt in tok for dt in detected_tokens))
    return round(min(10.0, (matches / len(prompt_tokens)) * 15), 1)


# ════════════════════════════════════════
# ENGINE: M2D — Music Audio Embedding (NTT Labs / HuggingFace)
# ════════════════════════════════════════
_m2d_model = None
_m2d_processor = None

def _load_m2d():
    """Lazy-load M2D model from HuggingFace (music-specific ViT, ~300MB)."""
    global _m2d_model, _m2d_processor
    if _m2d_model is None:
        import torch
        from transformers import AutoModel, AutoFeatureExtractor
        repo_id = "nttcslab-exp-010k/m2d_vit_base-80x608p16x16-221006-mr7"
        _m2d_processor = AutoFeatureExtractor.from_pretrained(repo_id, trust_remote_code=True)
        _m2d_model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
        _m2d_model.eval()
        print("[INFO] M2D model loaded successfully")
    return _m2d_model, _m2d_processor


def extract_m2d_embedding(audio_bytes: bytes) -> dict:
    """M2D: Extract 768-dim music audio embedding for cross-audio comparison."""
    import librosa as lr
    import torch

    try:
        model, processor = _load_m2d()
    except Exception as e:
        return {"engine": "m2d", "error": f"Model load failed: {e}"}

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        y, _ = lr.load(tmp_path, sr=16000, mono=True)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
        return {
            "engine": "m2d",
            "embedding_dim": len(embedding),
            "embedding": embedding,
        }
    except Exception as e:
        return {"engine": "m2d", "error": str(e)}
    finally:
        os.unlink(tmp_path)


def extract_clap_embedding_only(audio_bytes: bytes) -> list:
    """Extract raw 512-dim CLAP audio embedding. Reuses cached model."""
    import torch
    import librosa as lr
    try:
        model = _load_clap()
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            y, _ = lr.load(tmp_path, sr=48000, mono=True)
            audio_data = y.reshape(1, -1)
            with torch.no_grad():
                embed = model.get_audio_embedding_from_data(audio_data, use_tensor=True)
            return embed.cpu().squeeze().numpy().tolist()
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        print(f"[WARN] CLAP embedding extraction failed: {e}")
        return []


# ════════════════════════════════════════
# Gemini Embedding 2 — Text Embedding (upgraded)
# ════════════════════════════════════════
def _get_embedding_v2(text: str) -> list:
    """Get text embedding via Gemini Embedding 2 REST API (3072-dim, 8K context)."""
    if not text or not text.strip():
        return []
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-exp-03-07:embedContent"
        payload = {
            "model": "models/gemini-embedding-exp-03-07",
            "content": {"parts": [{"text": text.strip()}]}
        }
        with httpx.Client(timeout=15) as client:
            resp = client.post(url, json=payload, params={"key": GEMINI_KEY})
            resp.raise_for_status()
            return resp.json()["embedding"]["values"]
    except Exception as e:
        print(f"[WARN] Embedding v2 failed: {type(e).__name__}: {e}")
        return []


def _embedding_cosine_sim(vec_a: list, vec_b: list) -> float:
    """Cosine similarity between two embedding vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    a, b = np.array(vec_a), np.array(vec_b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


# ════════════════════════════════════════
# PROMPT PARSING HELPERS
# ════════════════════════════════════════
def _extract_bpm_from_prompt(prompt: str):
    """Extract BPM number from prompt text."""
    m = re.search(r'(\d{2,3})\s*bpm', prompt.lower())
    return int(m.group(1)) if m else None


def _extract_key_from_prompt(prompt: str):
    """Extract musical key from prompt text."""
    m = re.search(r'\b([A-G][#b]?)\s*(major|minor|maj|min)\b', prompt, re.IGNORECASE)
    if m:
        key = m.group(1)
        scale = "minor" if m.group(2).lower().startswith("min") else "major"
        return f"{key} {scale}"
    return None


def _calc_bpm_fidelity(target: int, detected: float) -> float:
    """BPM fidelity score (0-10), handles half/double BPM detection."""
    if not target or not detected:
        return 0.0
    ratio = min(target, detected) / max(target, detected)
    ratio_half = min(target, detected * 2) / max(target, detected * 2) if detected > 0 else 0
    ratio_dbl = min(target * 2, detected) / max(target * 2, detected) if target > 0 else 0
    return round(max(ratio, ratio_half, ratio_dbl) * 10, 1)


def _calc_key_fidelity(target: str, detected: str) -> float:
    """Key fidelity score (0-10)."""
    if not target or not detected:
        return 0.0
    if target.lower() == detected.lower():
        return 10.0
    # Same root, different mode = partial credit
    t_root = target.split()[0] if target else ""
    d_root = detected.split()[0] if detected else ""
    if t_root.lower() == d_root.lower():
        return 7.0
    # Enharmonic equivalents
    enharmonic = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}
    enharmonic.update({v: k for k, v in enharmonic.items()})
    if enharmonic.get(t_root, "") == d_root:
        return 10.0
    return 0.0


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


async def run_reverse_prompt(librosa_data: dict, essentia_data: dict, gemini_flash_data: str, few_shot_examples: list = None) -> dict:
    """Predict Suno prompt from audio analysis using Claude Sonnet 4.6.
    few_shot_examples: RAG 검색 결과 (유사 케이스 프롬프트 참고용)
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    prompt_text = REVERSE_PROMPT_TEMPLATE.format(
        librosa_data=json.dumps(librosa_data, indent=2),
        essentia_data=json.dumps(essentia_data, indent=2),
        gemini_data=gemini_flash_data
    )
    # RAG few-shot 주입
    if few_shot_examples:
        examples_lines = []
        for i, ex in enumerate(few_shot_examples[:3]):
            sim = ex.get("similarity", 0)
            prompt_str = ex.get("original_prompt", "")
            score = ex.get("overall_score")
            score_str = f", score={score}" if score is not None else ""
            examples_lines.append(f"  [{i+1}] (similarity={sim:.2f}{score_str}) \"{prompt_str}\"")
        few_shot_block = (
            "\n\n<reference_cases>\n"
            "Similar audio cases from history (use as style reference for token selection):\n"
            + "\n".join(examples_lines) +
            "\n</reference_cases>"
        )
        prompt_text = prompt_text + few_shot_block
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        messages=[{"role": "user", "content": prompt_text}]
    )
    raw = extract_json(response.content[0].text)
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
    """
    Quality score (0-10) with three scoring axes:
    1. Continuous value scaling (not just boolean existence)
    2. Cross-engine validation penalties (librosa vs essentia disagreement)
    3. Gemini analysis depth differentiation
    """
    score = 0.0

    # ── AXIS 1: Continuous value scoring (max ~5.0) ──

    # librosa: BPM presence + duration quality (max 1.5)
    if librosa_data.get("bpm") and librosa_data["bpm"] > 0:
        score += 0.5
        # Longer tracks = more reliable analysis
        dur = librosa_data.get("duration_seconds", 0)
        if dur > 120:
            score += 0.5
        elif dur > 60:
            score += 0.3
        elif dur > 30:
            score += 0.1
        # Dynamic range: wider = more interesting signal
        dr = librosa_data.get("dynamic_range_db", 0)
        if dr > 15:
            score += 0.5
        elif dr > 8:
            score += 0.3
        elif dr > 3:
            score += 0.1

    # essentia: key strength proportional (max 1.5)
    ks = essentia_data.get("key_strength", 0) or 0
    score += min(ks, 1.0) * 1.0  # 0~1.0 proportional
    if essentia_data.get("danceability") is not None and not essentia_data.get("error"):
        score += 0.5

    # gemini confidence proportional (max 2.0)
    if isinstance(gemini_flash, dict) and not gemini_flash.get("error"):
        conf = gemini_flash.get("analysis_confidence", 0) or 0
        score += min(conf / 10.0, 1.0) * 2.0  # conf 7→1.4, 8→1.6, 9→1.8, 10→2.0

    # ── AXIS 2: Cross-engine validation (max +2.0, with penalties) ──

    # BPM agreement: librosa vs essentia
    bpm_l = librosa_data.get("bpm", 0) or 0
    bpm_e = essentia_data.get("bpm", 0) or 0
    if bpm_l > 0 and bpm_e > 0:
        bpm_ratio = min(bpm_l, bpm_e) / max(bpm_l, bpm_e)
        # Also check double/half BPM (common detection discrepancy)
        bpm_ratio_half = min(bpm_l, bpm_e * 2) / max(bpm_l, bpm_e * 2) if bpm_e > 0 else 0
        bpm_ratio_dbl = min(bpm_l * 2, bpm_e) / max(bpm_l * 2, bpm_e) if bpm_l > 0 else 0
        best_ratio = max(bpm_ratio, bpm_ratio_half, bpm_ratio_dbl)
        if best_ratio > 0.95:
            score += 1.0  # Near-perfect agreement
        elif best_ratio > 0.85:
            score += 0.5  # Close enough
        else:
            score -= 0.5  # Significant disagreement = penalty

    # Key agreement: librosa vs essentia
    key_l = librosa_data.get("key", "")
    key_e = essentia_data.get("key", "")
    scale_l = librosa_data.get("scale", "")
    scale_e = essentia_data.get("scale", "")
    if key_l and key_e:
        # Relative major/minor pairs (C major ↔ A minor, etc.)
        relative_pairs = {
            "C": "A", "G": "E", "D": "B", "A": "F#", "E": "C#", "B": "G#",
            "F": "D", "Bb": "G", "Eb": "C", "Ab": "F", "Db": "Bb", "Gb": "Eb",
            "F#": "D#", "C#": "A#"
        }
        same_key = (key_l == key_e)
        relative_match = False
        if not same_key:
            # Check relative major/minor relationship
            if scale_l == "major" and scale_e == "minor":
                relative_match = relative_pairs.get(key_l) == key_e
            elif scale_l == "minor" and scale_e == "major":
                relative_match = relative_pairs.get(key_e) == key_l
        if same_key and scale_l == scale_e:
            score += 1.0  # Exact match
        elif same_key or relative_match:
            score += 0.5  # Partial match (same root or relative key)
        else:
            score -= 0.3  # Key disagreement = mild penalty

    # ── AXIS 3: Gemini analysis depth (max +3.0) ──

    if isinstance(gemini_flash, dict) and not gemini_flash.get("error"):
        genre_str = gemini_flash.get("genre", "") or ""
        mood_str = gemini_flash.get("mood", "") or ""
        instruments = gemini_flash.get("instruments", []) or []
        vocal = gemini_flash.get("vocal_type", "") or ""
        prod = gemini_flash.get("production_style", "") or ""

        # Genre specificity: multi-word/fusion genres score higher (max 1.0)
        genre_tokens = [t.strip() for t in genre_str.replace("/", ",").replace("-", " ").split(",") if t.strip()]
        if len(genre_tokens) >= 3:
            score += 1.0
        elif len(genre_tokens) == 2:
            score += 0.7
        elif len(genre_tokens) == 1 and genre_str.lower() not in ["unknown", "error", ""]:
            score += 0.3

        # Instrument detail (max 1.0)
        n_inst = len(instruments)
        if n_inst >= 4:
            score += 1.0
        elif n_inst >= 2:
            score += 0.6
        elif n_inst == 1:
            score += 0.2

        # Analysis completeness: vocal + production + mood all present (max 1.0)
        completeness = sum([
            1 if vocal and vocal.lower() not in ["none", "n/a", ""] else 0,
            1 if prod and prod.lower() not in ["unknown", ""] else 0,
            1 if len(mood_str.split(",")) >= 2 else 0,
        ])
        score += completeness / 3.0 * 1.0

    return round(max(0.0, min(score, 10.0)), 1)


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


async def find_similar_by_m2d(embedding: list, top_k: int = 3) -> list:
    """pgvector HNSW 검색: M2D 임베딩 기준 유사 케이스 top-k 반환."""
    if not embedding:
        return []
    try:
        vec_str = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/find_similar_audio",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json"
                },
                json={"query_embedding": vec_str, "match_count": top_k}
            )
        if r.status_code < 300:
            return r.json()
        return []
    except Exception as e:
        print(f"[WARN] RAG lookup failed: {e}")
        return []


# ════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "suno-audio-analyzer",
        "version": "7.2-rag",
        "modes": {
            "mode_a": "Prompt + Audio → Quad Engine → Claude Opus evaluation",
            "mode_b": "Audio only → librosa + Essentia + Gemini Flash + CLAP tagger → Claude Sonnet reverse prompt",
            "fidelity": "Prompt + Audio → CLAP + CLAP-tagger(MTT) + M2D + librosa + Essentia + Gemini Flash + Gemini Embedding 2 + Claude Sonnet → Multi-dim fidelity"
        },
        "engines": {
            "lite": ["librosa", "essentia", GEMINI_FLASH_MODEL],
            "pro": ["librosa", "essentia", GEMINI_PRO_MODEL, "gpt-4o-audio", "claude-opus-4-6"],
            "fidelity": ["clap-htsat", "clap-tagger-mtt50", "m2d-vit", "librosa", "essentia", GEMINI_FLASH_MODEL, "gemini-embedding-2", "claude-sonnet-4-6"],
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
    ip: str = Form("unknown"),
    model_version: str = Form("v5")
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
                "analysis_mode": "prompt_eval",
                "model_version": model_version
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

            # Engine 4: CLAP music tagger (MTT zero-shot) + embedding 추출 병렬 실행
            loop = asyncio.get_event_loop()
            try:
                clap_tagger_result, clap_embed = await asyncio.gather(
                    loop.run_in_executor(None, score_clap_music_tagger, audio_bytes, ""),
                    loop.run_in_executor(None, extract_clap_embedding_only, audio_bytes),
                )
            except Exception as e:
                clap_tagger_result = {"engine": "clap_tagger", "error": str(e)}
                clap_embed = []

            # Engine 5: M2D embedding 추출
            try:
                m2d_result = await loop.run_in_executor(None, extract_m2d_embedding, audio_bytes)
                m2d_embed = m2d_result.get("embedding", [])
            except Exception as e:
                m2d_result = {"engine": "m2d", "error": str(e)}
                m2d_embed = []

            # RAG: 유사 케이스 검색 (임베딩 있을 때만)
            few_shot_examples = []
            if m2d_embed:
                try:
                    few_shot_examples = await find_similar_by_m2d(m2d_embed, top_k=3)
                except Exception:
                    few_shot_examples = []

            # Reverse prompt prediction (+ RAG few-shot)
            try:
                predicted = await run_reverse_prompt(
                    librosa_result, essentia_result, flash_report_str,
                    few_shot_examples=few_shot_examples
                )
            except Exception as e:
                predicted = {"predicted_prompt": "", "confidence": 0, "error": str(e)}

            # Data quality score
            quality_score = compute_data_quality_score(librosa_result, essentia_result, flash_report)
            reeval_recommended = quality_score >= REEVAL_SCORE_THRESHOLD

            # Save to Supabase (벡터 콼럼 포함)
            db_result = {
                "ip": ip,
                "original_prompt": predicted.get("predicted_prompt", ""),
                "gemini_report": json.dumps({
                    "librosa": librosa_result,
                    "essentia": essentia_result,
                    "gemini_flash": flash_report,
                    "clap_tagger": clap_tagger_result,
                    "m2d": {"engine": "m2d", "embedding_dim": len(m2d_embed)},
                    "predicted": predicted
                }),
                "analysis_mode": "audio_only",
                "prompt_type": "predicted",
                "overall_score": None,
                "quality_score": round(quality_score, 2),
            }
            if m2d_embed:
                db_result["m2d_embedding"] = "[" + ",".join(f"{v:.6f}" for v in m2d_embed) + "]"
            if clap_embed:
                db_result["clap_embedding"] = "[" + ",".join(f"{v:.6f}" for v in clap_embed) + "]"
            saved = await save_to_supabase(db_result)
            saved_id = saved[0].get("id") if saved and isinstance(saved, list) else None

            return {
                "mode": "audio_only",
                "librosa": librosa_result,
                "essentia": essentia_result,
                "gemini_flash": flash_report,
                "clap_tagger": clap_tagger_result,
                "predicted_prompt": predicted,
                "quality_score": quality_score,
                "reeval_recommended": reeval_recommended,
                "rag_similar_count": len(few_shot_examples),
                "analysis_id": saved_id,
                "label": label
            }

    except asyncio.TimeoutError:
        raise HTTPException(503, "Server busy. Please try again in a moment.")


# ────────────────────────────────────────
# MODE C: FIDELITY CHECK — Prompt↔Audio Multi-Engine Comparison
# CLAP + PANNs + librosa + Essentia + Gemini Embedding 2
# ────────────────────────────────────────
@app.post("/fidelity")
async def fidelity_check(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    ip: str = Form("unknown")
):
    """
    Comprehensive fidelity check: How well does the generated audio match the original prompt?
    Uses 6 engines: CLAP (text↔audio), PANNs (audio tagging), librosa, Essentia,
    Gemini Flash (subjective), Gemini Embedding 2 (text similarity).
    Returns multi-dimensional fidelity scores + composite score.
    """
    if not file.content_type or "audio" not in file.content_type:
        raise HTTPException(400, "File must be an audio file (mp3, wav, etc.)")

    audio_bytes = await file.read()
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(400, "File too large. Maximum 25MB.")

    try:
        async with analysis_semaphore:
            loop = asyncio.get_event_loop()

            # ── Parallel engine execution ──
            # CPU-bound engines in thread pool
            librosa_future = loop.run_in_executor(None, analyze_with_librosa, audio_bytes)
            essentia_future = loop.run_in_executor(None, analyze_with_essentia, audio_bytes)
            clap_future = loop.run_in_executor(None, get_clap_alignment, audio_bytes, prompt)
            clap_tagger_future = loop.run_in_executor(None, score_clap_music_tagger, audio_bytes, prompt)
            m2d_future = loop.run_in_executor(None, extract_m2d_embedding, audio_bytes)

            # Wait for all CPU engines
            librosa_result = await librosa_future
            essentia_result = await essentia_future
            clap_result = await clap_future
            clap_tagger_result = await clap_tagger_future
            m2d_result = await m2d_future

            # Async engine: Gemini Flash (needs audio bytes)
            try:
                flash_raw = await run_gemini_flash(audio_bytes, file.content_type or "audio/mpeg")
                flash_str = extract_json(flash_raw)
                flash_report = json.loads(flash_str) if flash_str.startswith("{") else {"raw": flash_str}
            except Exception as e:
                flash_report = {"error": str(e)}
                flash_str = json.dumps(flash_report)

            # Reverse prompt prediction
            try:
                predicted = await run_reverse_prompt(librosa_result, essentia_result, flash_str)
            except Exception as e:
                predicted = {"predicted_prompt": "", "confidence": 0, "error": str(e)}

            # Gemini Embedding 2: prompt text vs predicted prompt text
            predicted_prompt_text = predicted.get("predicted_prompt", "")
            prompt_embed = _get_embedding_v2(prompt)
            predicted_embed = _get_embedding_v2(predicted_prompt_text)
            text_sim = _embedding_cosine_sim(prompt_embed, predicted_embed)

            # ── Fidelity Dimension Scores (all 0-10 scale) ──
            dimensions = {}

            # 1. CLAP alignment (direct text↔audio)
            clap_score = clap_result.get("alignment_score_10")
            if clap_score is not None:
                dimensions["clap_alignment"] = clap_score

            # 2. BPM fidelity
            bpm_target = _extract_bpm_from_prompt(prompt)
            bpm_detected = librosa_result.get("bpm", 0)
            if bpm_target:
                dimensions["bpm_fidelity"] = _calc_bpm_fidelity(bpm_target, bpm_detected)

            # 3. Key fidelity
            key_target = _extract_key_from_prompt(prompt)
            key_detected = librosa_result.get("key_full", "")
            if key_target:
                dimensions["key_fidelity"] = _calc_key_fidelity(key_target, key_detected)

            # 4. Text embedding similarity (prompt vs reverse-predicted prompt)
            if text_sim > 0:
                normalized_text = max(0.0, min(1.0, (text_sim - 0.5) / 0.35)) * 10
                dimensions["text_similarity"] = round(normalized_text, 1)

            # 5. Gemini Flash tag match (genre/mood/instrument keywords vs prompt)
            tag_score = score_gemini_tag_match(prompt, flash_report)
            if tag_score > 0:
                dimensions["gemini_tag_match"] = tag_score

            # 6. CLAP music tagger (MTT vocabulary zero-shot — MusiCNN equivalent)
            clap_tagger_score = clap_tagger_result.get("tag_match_score")
            if clap_tagger_score is not None and not clap_tagger_result.get("error"):
                dimensions["clap_tagger_match"] = clap_tagger_score

            # ── Composite Fidelity Score (weighted) ──
            weights = {
                "clap_alignment": 3.0,      # Most important: direct text↔audio
                "bpm_fidelity": 1.5,
                "key_fidelity": 2.0,
                "text_similarity": 2.0,
                "gemini_tag_match": 1.5,    # Gemini Flash semantic tags vs prompt
                "clap_tagger_match": 1.0,   # CLAP MTT zero-shot tags vs prompt (MusiCNN equiv)
            }
            if dimensions:
                total_w = sum(weights.get(k, 1.0) for k in dimensions)
                composite = sum(dimensions[k] * weights.get(k, 1.0) for k in dimensions) / total_w
            else:
                composite = 0.0

            # ── Save to Supabase ──
            db_result = {
                "ip": ip,
                "original_prompt": prompt,
                "gemini_report": json.dumps({
                    "librosa": librosa_result,
                    "essentia": essentia_result,
                    "clap": clap_result,
                    "clap_tagger": clap_tagger_result,
                    "m2d": {"embedding_dim": m2d_result.get("embedding_dim"), "error": m2d_result.get("error")},
                    "m2d_embedding": m2d_result.get("embedding"),  # stored for /compare
                    "gemini_flash": flash_report,
                    "predicted": predicted,
                    "text_embedding_similarity": round(text_sim, 4),
                    "fidelity_dimensions": dimensions,
                }),
                "analysis_mode": "fidelity_check",
                "prompt_type": "style",
                "overall_score": round(composite, 1),
            }
            saved = await save_to_supabase(db_result)
            saved_id = saved[0].get("id") if saved and isinstance(saved, list) else None

            return {
                "mode": "fidelity_check",
                "composite_fidelity": round(composite, 1),
                "dimensions": dimensions,
                "clap": clap_result,
                "m2d": {"embedding_dim": m2d_result.get("embedding_dim"), "error": m2d_result.get("error")},
                "librosa": librosa_result,
                "essentia": essentia_result,
                "gemini_flash": flash_report,
                "predicted_prompt": predicted,
                "text_embedding_similarity": round(text_sim, 4),
                "prompt_analysis": {
                    "bpm_requested": bpm_target,
                    "key_requested": key_target,
                    "bpm_detected": bpm_detected,
                    "key_detected": key_detected,
                },
                "analysis_id": saved_id,
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


# ────────────────────────────────────────
# BATCH: Server-side CDN download + Mode A analyze
# ────────────────────────────────────────
class BatchTrack(BaseModel):
    id: str
    audio_url: str
    style_prompt: str
    model_version: str = "v5"
    title: str = ""

class BatchRequest(BaseModel):
    tracks: list[BatchTrack]
    suno_token: str
    admin_key: str = ""

@app.post("/batch-analyze")
async def batch_analyze(req: BatchRequest):
    """Server-side batch: Railway downloads from CDN + runs Mode A analysis."""
    if ADMIN_KEY and req.admin_key != ADMIN_KEY:
        raise HTTPException(403, "Invalid admin_key")

    results = []
    sem = asyncio.Semaphore(1)

    async def process_one(track: BatchTrack):
        async with sem:
            try:
                async with httpx.AsyncClient(timeout=60) as dl:
                    r = await dl.get(
                        track.audio_url,
                        headers={"Authorization": req.suno_token},
                        follow_redirects=True
                    )
                    if r.status_code != 200:
                        results.append({"id": track.id, "status": "dl_failed", "code": r.status_code})
                        return
                    audio_bytes = r.content

                try:
                    librosa_result = analyze_with_librosa(audio_bytes)
                except Exception as e:
                    librosa_result = {"engine": "librosa", "error": str(e)}
                try:
                    essentia_result = analyze_with_essentia(audio_bytes)
                except Exception as e:
                    essentia_result = {"engine": "essentia", "error": str(e)}
                try:
                    gemini_raw = await run_gemini(audio_bytes, "audio/mpeg")
                    gemini_report = extract_json(gemini_raw)
                except Exception as e:
                    gemini_report = json.dumps({"engine": "gemini", "error": str(e)})
                try:
                    gpt4o_raw = await run_gpt4o(audio_bytes, "audio/mpeg")
                    gpt4o_report = extract_json(gpt4o_raw)
                except Exception as e:
                    gpt4o_report = json.dumps({"engine": "gpt4o", "error": str(e)})
                try:
                    claude_eval = await run_claude(track.style_prompt, librosa_result, essentia_result, gemini_report, gpt4o_report)
                except Exception as e:
                    claude_eval = {"overall_score": None, "summary": str(e)}

                full_report = json.dumps({
                    "librosa": librosa_result,
                    "essentia": essentia_result,
                    "gemini": json.loads(gemini_report) if gemini_report.startswith("{") else gemini_report,
                    "gpt4o": json.loads(gpt4o_report) if gpt4o_report.startswith("{") else gpt4o_report
                })
                db_result = {
                    "original_prompt": track.style_prompt,
                    "gemini_report": full_report,
                    "genre_accuracy": claude_eval.get("genre_accuracy"),
                    "bpm_accuracy": claude_eval.get("bpm_accuracy"),
                    "instrument_accuracy": claude_eval.get("instrument_accuracy"),
                    "mood_accuracy": claude_eval.get("mood_accuracy"),
                    "structure_accuracy": claude_eval.get("structure_accuracy"),
                    "overall_score": claude_eval.get("overall_score"),
                    "prompt_type": "style",
                    "analysis_mode": "prompt_eval",
                    "model_version": track.model_version
                }
                await save_to_supabase(db_result)
                results.append({"id": track.id, "status": "ok", "overall_score": claude_eval.get("overall_score")})
            except Exception as e:
                results.append({"id": track.id, "status": "error", "error": str(e)})
            await asyncio.sleep(2)

    await asyncio.gather(*[process_one(t) for t in req.tracks])
    ok = sum(1 for r in results if r.get("status") == "ok")
    return {"total": len(req.tracks), "success": ok, "results": results}

