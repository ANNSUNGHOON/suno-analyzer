from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import anthropic
import httpx
import json
import os
import tempfile
import re

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

GEMINI_ANALYSIS_PROMPT = """You are an expert music production analyst. Analyze this audio file in detail.

Respond ONLY in valid JSON format with NO other text:
{
  "genre": "detected genre(s)",
  "bpm_estimate": "estimated BPM range",
  "instruments": ["list", "of", "detected", "instruments"],
  "mood": "overall mood/atmosphere description",
  "structure": "song structure description (intro, verse, chorus, etc.)",
  "production_notes": "production quality, mixing characteristics, sound design details",
  "vocal_type": "vocal characteristics if present, or 'instrumental'",
  "frequency_balance": "description of low/mid/high frequency balance",
  "stereo_field": "stereo width and spatial characteristics",
  "dynamics": "dynamic range and energy flow description"
}"""

CLAUDE_EVALUATION_PROMPT = """You are evaluating how accurately an AI music generator (Suno) interpreted a style prompt.

ORIGINAL PROMPT given to Suno:
{prompt}

AUDIO ANALYSIS REPORT from an independent AI analyzer:
{report}

Rate the accuracy of the generated music compared to the original prompt intent on each dimension (1-10 scale):

Respond ONLY in valid JSON:
{{
  "genre_accuracy": <1-10>,
  "bpm_accuracy": <1-10>,
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
    """Send audio to Gemini for analysis."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    response = model.generate_content([
        GEMINI_ANALYSIS_PROMPT,
        {"mime_type": mime_type, "data": audio_bytes}
    ])
    return response.text


async def evaluate_with_claude(original_prompt: str, gemini_report: str) -> dict:
    """Send Gemini report + original prompt to Claude for evaluation."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": CLAUDE_EVALUATION_PROMPT.format(
                prompt=original_prompt,
                report=gemini_report
            )
        }]
    )
    
    text = message.content[0].text
    # Clean JSON from markdown fences if present
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
    return {"status": "ok", "service": "suno-audio-analyzer"}


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
    if len(audio_bytes) > 25 * 1024 * 1024:  # 25MB limit
        raise HTTPException(400, "File too large. Maximum 25MB.")
    
    # Step 1: Gemini analyzes the audio
    try:
        gemini_raw = await analyze_with_gemini(audio_bytes, file.content_type)
        gemini_report = extract_json(gemini_raw)
    except Exception as e:
        raise HTTPException(500, f"Gemini analysis failed: {str(e)}")
    
    # Step 2: Claude evaluates prompt vs result
    try:
        claude_eval = await evaluate_with_claude(prompt, gemini_report)
    except Exception as e:
        # If Claude fails, still save Gemini report
        claude_eval = {
            "genre_accuracy": None,
            "bpm_accuracy": None,
            "instrument_accuracy": None,
            "mood_accuracy": None,
            "structure_accuracy": None,
            "overall_score": None,
            "summary": f"Claude evaluation failed: {str(e)}",
            "token_feedback": []
        }
    
    # Step 3: Save to Supabase
    result = {
        "ip": ip,
        "prompt_id": prompt_id,
        "original_prompt": prompt,
        "gemini_report": gemini_report,
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
        "gemini_report": json.loads(gemini_report) if gemini_report.startswith("{") else gemini_report,
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
    """Analyze Suno song by URL - attempts to extract audio."""
    # Try to get the actual audio URL from Suno
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            # Try direct CDN URL pattern
            # Suno audio is typically at cdn1.suno.ai or cdn2.suno.ai
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
    
    # Same pipeline as upload
    try:
        gemini_raw = await analyze_with_gemini(audio_bytes)
        gemini_report = extract_json(gemini_raw)
    except Exception as e:
        raise HTTPException(500, f"Gemini analysis failed: {str(e)}")
    
    try:
        claude_eval = await evaluate_with_claude(prompt, gemini_report)
    except Exception as e:
        claude_eval = {
            "genre_accuracy": None, "bpm_accuracy": None,
            "instrument_accuracy": None, "mood_accuracy": None,
            "structure_accuracy": None, "overall_score": None,
            "summary": f"Claude evaluation failed: {str(e)}",
            "token_feedback": []
        }
    
    result = {
        "ip": ip, "prompt_id": prompt_id,
        "original_prompt": prompt, "gemini_report": gemini_report,
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
        "gemini_report": json.loads(gemini_report) if gemini_report.startswith("{") else gemini_report,
        "evaluation": claude_eval,
        "saved": True
    }
