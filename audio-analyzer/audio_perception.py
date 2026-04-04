"""Audio Perception Layer (Layer 2) — LLM-based closed-label extraction.

Abstracts API LLM clients for prism_vector_v1 extraction.
Supports Gemini 2.5 Flash (primary), Qwen3 (fallback), and Qwen 3.5 Omni Plus (eval).

Usage:
    from audio_perception import create_client
    client = create_client("gemini")  # or "qwen", "qwen35plus"
    vector = client.extract("path/to/audio.mp3", signal_metrics)
"""

from __future__ import annotations

import base64
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from .prism_schema import (
        PrismVector, ObjectiveMetrics, get_allowed_labels,
    )
except ImportError:
    from prism_schema import (  # type: ignore
        PrismVector, ObjectiveMetrics, get_allowed_labels,
    )


# ──────────────────────────────────────────────────────
# Shared prompt templates
# ──────────────────────────────────────────────────────

EXTRACTOR_SYSTEM = """You are Prism-Extractor for Music_Jarvis.

Your only job:
- Listen to the audio.
- Read the objective Signal Engine metrics.
- Fill a CLOSED LABEL VECTOR.
- Return JSON ONLY.
- No markdown. No prose. No artist names.
- No lyrical / religious / narrative interpretation.

Critical rule:
Do not invent emotional metaphors such as prayer flow, emotional climax, spiritual presence, message delivery.
Only classify audible production traits.
If uncertain, lower confidence. Do not compensate with vague language.
"""


def build_extractor_prompt(metrics: ObjectiveMetrics) -> str:
    """Build the extractor user prompt with allowed labels and metrics."""
    labels = get_allowed_labels()

    def fmt(key: str) -> str:
        return json.dumps(sorted(labels[key]), ensure_ascii=False)

    return f"""Return ONE valid JSON object matching this exact structure and only using allowed labels.

Allowed labels:

vocal_presence = {fmt("vocal_presence")}
lead_focus = {fmt("lead_focus")}
vocal_attack = {fmt("vocal_attack")}
vocal_position = {fmt("vocal_position")}
vocal_texture = {fmt("vocal_texture")}

low_mid_density = {fmt("low_mid_density")}
mid_focus = {fmt("mid_focus")}
top_end_presence = {fmt("top_end_presence")}
brightness = {fmt("brightness")}
reverb_tail = {fmt("reverb_tail")}
stereo_width = {fmt("stereo_width")}
pad_spread = {fmt("pad_spread")}

percussion_presence = {fmt("percussion_presence")}
rhythm_driver = {fmt("rhythm_driver")}
groove_motion = {fmt("groove_motion")}

dynamic_shape = {fmt("dynamic_shape")}
lift_method = {fmt("lift_method")}

main_anchor = {fmt("main_anchor")}
secondary_shading = {fmt("secondary_shading")}
preserve_tags = {fmt("preserve_tags")}
avoid_tags = {fmt("avoid_tags")}

Signal Engine metrics:
- key = {metrics.key}
- scale = {metrics.scale}
- key_strength = {metrics.key_strength}
- bpm = {metrics.bpm}
- spectral_centroid = {metrics.spectral_centroid}
- energy_normalized = {metrics.energy_normalized}

JSON shape:
{{
  "schema_version": "prism_vector_v1",
  "analysis": {{
    "vocal": {{ "vocal_presence": "...", "lead_focus": "...", "vocal_attack": "...", "vocal_position": "...", "vocal_texture": ["..."], "confidence": 0.0 }},
    "tone": {{ "low_mid_density": "...", "mid_focus": "...", "top_end_presence": "...", "brightness": "...", "reverb_tail": "...", "stereo_width": "...", "pad_spread": "...", "confidence": 0.0 }},
    "rhythm": {{ "percussion_presence": "...", "rhythm_driver": "...", "groove_motion": "...", "confidence": 0.0 }},
    "arrangement": {{ "dynamic_shape": "...", "lift_method": "...", "confidence": 0.0 }},
    "genre": {{ "main_anchor": "...", "secondary_shading": ["..."], "confidence": 0.0 }}
  }},
  "prompt_transfer": {{
    "preserve_tags": ["..."],
    "avoid_tags": ["..."],
    "ambiguity_note": "short practical note"
  }}
}}

Rules:
- JSON only, no source field (Python fills it)
- confidence values must be between 0 and 1
- keep lists short and practical
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract first JSON object from LLM text output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start:end + 1])


def _load_audio_b64(path: Path) -> str:
    """Load audio file as base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ──────────────────────────────────────────────────────
# Abstract client
# ──────────────────────────────────────────────────────

class AudioPerceptionClient(ABC):
    """Abstract base for Audio Perception Layer clients."""

    @abstractmethod
    def extract_raw(
        self, audio_path: str, metrics: ObjectiveMetrics
    ) -> dict[str, Any]:
        """Send audio + metrics to LLM, return raw parsed JSON dict."""
        ...

    def extract(
        self, audio_path: str, metrics: ObjectiveMetrics
    ) -> PrismVector:
        """Full extraction: LLM call → parse → validate → inject source."""
        raw = self.extract_raw(audio_path, metrics)
        audio_file = Path(audio_path).name
        return PrismVector.from_llm_output(raw, audio_file, metrics)


# ──────────────────────────────────────────────────────
# Gemini 2.5 Flash client
# ──────────────────────────────────────────────────────

class GeminiClient(AudioPerceptionClient):
    """Gemini 2.5 Flash — primary Layer 2 model.

    Uses google-generativeai SDK with native JSON schema enforcement.
    ~$0.002/track, ~8.4hr audio limit.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model
        self._api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY not set")

        import google.generativeai as genai
        genai.configure(api_key=self._api_key)
        self._genai = genai

    def extract_raw(
        self, audio_path: str, metrics: ObjectiveMetrics
    ) -> dict[str, Any]:
        prompt = build_extractor_prompt(metrics)

        # Upload audio file
        audio_file = self._genai.upload_file(audio_path)
        logger.info(f"Gemini: uploaded {audio_path} -> {audio_file.uri}")

        model = self._genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=EXTRACTOR_SYSTEM,
            generation_config=self._genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        response = model.generate_content([audio_file, prompt])
        raw_text = response.text
        logger.info(f"Gemini: got {len(raw_text)} chars response")

        return _extract_json_object(raw_text)


# ──────────────────────────────────────────────────────
# Qwen3 client (DashScope)
# ──────────────────────────────────────────────────────

class QwenClient(AudioPerceptionClient):
    """Qwen3-Omni-Flash via DashScope OpenAI-compatible API.

    Fallback model. ~$0.02/track (international).
    """

    def __init__(
        self,
        model: str = "qwen3-omni-flash",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        self.model_name = model
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError("DASHSCOPE_API_KEY not set")

        from openai import OpenAI
        self._client = OpenAI(api_key=self._api_key, base_url=base_url)

    def extract_raw(
        self, audio_path: str, metrics: ObjectiveMetrics
    ) -> dict[str, Any]:
        prompt = build_extractor_prompt(metrics)
        audio_b64 = _load_audio_b64(Path(audio_path))
        audio_data_url = f"data:;base64,{audio_b64}"

        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": EXTRACTOR_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_data_url, "format": "wav"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            modalities=["text"],
            stream=False,
        )

        raw_text = completion.choices[0].message.content
        logger.info(f"Qwen: got {len(raw_text)} chars response")

        return _extract_json_object(raw_text)


# ──────────────────────────────────────────────────────
# Qwen 3.5 Omni Plus client (DashScope)
# ──────────────────────────────────────────────────────

class Qwen35PlusClient(QwenClient):
    """Qwen 3.5 Omni Plus via DashScope OpenAI-compatible API.

    Evaluation candidate. Same DashScope infra as QwenClient,
    model ID only change (IL-4 single-variable).
    Benchmarks: MuchoMusic 72.4 vs Gemini 3.1 Pro 59.6.
    """

    def __init__(
        self,
        model: str = "qwen3.5-omni-plus",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url)


# ──────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────

def create_client(
    provider: str = "gemini",
    **kwargs: Any,
) -> AudioPerceptionClient:
    """Create an Audio Perception client.

    Args:
        provider: "gemini" (default), "qwen", or "qwen35plus"
        **kwargs: Passed to client constructor.
    """
    providers = {
        "gemini": GeminiClient,
        "qwen": QwenClient,
        "qwen35plus": Qwen35PlusClient,
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(providers)}")
    return providers[provider](**kwargs)
