"""prism_vector_v1 schema — Pydantic v2 models for Audio Perception Layer (Layer 2).

Closed-label vector: 16 analysis dimensions + prompt_transfer.
All fields use Literal types to structurally prevent LLM hallucination.

Produced by: Audio Perception Layer (Gemini 2.5 Flash / Qwen3)
Consumed by: Prism (Layer 3), Signal Engine v2 integration
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ──────────────────────────────────────────────────────
# Enum literals — single source of truth
# ──────────────────────────────────────────────────────

VocalPresence = Literal["none", "low", "medium", "high"]
LeadFocus = Literal["vocal_led", "instrument_led", "balanced"]
VocalAttack = Literal["soft", "neutral", "hard"]
VocalPosition = Literal["front", "mid", "back"]
VocalTexture = Literal[
    "clean", "rounded", "breathy", "airy", "plain",
    "intimate", "gentle", "firm",
]

LowMidDensity = Literal["low", "medium", "high"]
MidFocus = Literal["low", "medium", "high"]
TopEndPresence = Literal["low", "medium", "high"]
Brightness = Literal["dark", "neutral", "bright"]
ReverbTail = Literal["short", "medium", "long"]
StereoWidth = Literal["narrow", "medium", "wide"]
PadSpread = Literal["none", "low", "medium", "high"]

PercussionPresence = Literal["none", "low", "medium", "high"]
RhythmDriver = Literal[
    "vocal_phrase", "piano_pattern", "drums", "bass", "pad_pulse", "mixed",
]
GrooveMotion = Literal["static", "gentle_push", "rolling", "pulsed"]

DynamicShape = Literal["flat", "gradual_build", "stepwise_build", "peak_and_release"]
LiftMethod = Literal[
    "none", "layer_addition", "density_rise", "space_expansion",
    "drum_entry", "hybrid",
]

MainAnchor = Literal[
    "ccm_ballad", "studio_worship", "soft_pop_ballad", "piano_ballad",
    "ambient_worship", "orchestral_ballad", "acoustic_ballad", "indie_ballad",
    "cinematic_ballad", "soft_rnb", "dark_electronic", "pop_dance",
    "rock", "rap_bass", "orchestral", "indie",
]

SecondaryShading = Literal[
    "studio_polish", "live_worship_hint", "soft_pop", "ambient_pad",
    "piano_led", "orchestral_lift", "restrained_dynamic", "mid_focused_tone",
    "warm_low_mid", "light_percussion", "gentle_build", "wide_space",
]

PreserveTag = Literal[
    "soft_vocal_attack", "front_vocal", "mid_vocal", "rounded_vocal_tone",
    "breathy_vocal_hint", "low_mid_warmth", "mid_focused_body",
    "controlled_top_end", "airy_top_edge", "long_reverb_tail",
    "wide_stereo_pad", "pad_backfill", "low_percussion_presence",
    "piano_led_timing", "vocal_led_timing", "gradual_layer_build",
    "space_expansion_lift", "restrained_dynamics", "studio_polish",
]

AvoidTag = Literal[
    "hard_drum_led_groove", "aggressive_transient_front", "over_bright_top_end",
    "dry_close_mix", "narrow_stereo_field", "heavy_kick_drive",
    "dense_percussion_grid", "abrupt_energy_jump",
    "overwritten_emotion_words", "lyric_meaning_overfit",
]


# ──────────────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────────────

class ObjectiveMetrics(BaseModel):
    """Signal Engine L1 metrics injected by Python post-processing."""
    model_config = ConfigDict(frozen=False)

    key: str = ""
    scale: str = ""
    key_strength: float = 0.0
    bpm: float = 0.0
    spectral_centroid: float = 0.0
    energy_normalized: float = 0.0


class PrismSource(BaseModel):
    """Metadata about the analyzed track — filled by Python, not LLM."""
    model_config = ConfigDict(frozen=False)

    audio_file: str = ""
    objective_metrics: ObjectiveMetrics = Field(default_factory=ObjectiveMetrics)


class VocalAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)

    vocal_presence: VocalPresence = "none"
    lead_focus: LeadFocus = "balanced"
    vocal_attack: VocalAttack = "neutral"
    vocal_position: VocalPosition = "mid"
    vocal_texture: list[VocalTexture] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ToneAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)

    low_mid_density: LowMidDensity = "medium"
    mid_focus: MidFocus = "medium"
    top_end_presence: TopEndPresence = "medium"
    brightness: Brightness = "neutral"
    reverb_tail: ReverbTail = "medium"
    stereo_width: StereoWidth = "medium"
    pad_spread: PadSpread = "none"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RhythmAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)

    percussion_presence: PercussionPresence = "none"
    rhythm_driver: RhythmDriver = "mixed"
    groove_motion: GrooveMotion = "static"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ArrangementAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)

    dynamic_shape: DynamicShape = "flat"
    lift_method: LiftMethod = "none"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GenreAnalysis(BaseModel):
    model_config = ConfigDict(frozen=False)

    main_anchor: MainAnchor = "soft_pop_ballad"
    secondary_shading: list[SecondaryShading] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class AnalysisBlock(BaseModel):
    model_config = ConfigDict(frozen=False)

    vocal: VocalAnalysis = Field(default_factory=VocalAnalysis)
    tone: ToneAnalysis = Field(default_factory=ToneAnalysis)
    rhythm: RhythmAnalysis = Field(default_factory=RhythmAnalysis)
    arrangement: ArrangementAnalysis = Field(default_factory=ArrangementAnalysis)
    genre: GenreAnalysis = Field(default_factory=GenreAnalysis)


class PromptTransfer(BaseModel):
    model_config = ConfigDict(frozen=False)

    preserve_tags: list[PreserveTag] = Field(default_factory=list)
    avoid_tags: list[AvoidTag] = Field(default_factory=list)
    ambiguity_note: str = ""


# ──────────────────────────────────────────────────────
# Top-level vector
# ──────────────────────────────────────────────────────

class PrismVector(BaseModel):
    """prism_vector_v1 — closed-label vector for Audio Perception Layer."""
    model_config = ConfigDict(frozen=False)

    schema_version: str = "prism_vector_v1"
    source: PrismSource = Field(default_factory=PrismSource)
    analysis: AnalysisBlock = Field(default_factory=AnalysisBlock)
    prompt_transfer: PromptTransfer = Field(default_factory=PromptTransfer)

    def inject_source(self, audio_file: str, metrics: ObjectiveMetrics) -> None:
        """Post-processing: inject Python-side metadata (not LLM-generated)."""
        self.source.audio_file = audio_file
        self.source.objective_metrics = metrics

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> PrismVector:
        return cls.model_validate(data)

    @classmethod
    def from_llm_output(
        cls, raw: dict, audio_file: str, metrics: ObjectiveMetrics,
        strict: bool = False,
    ) -> PrismVector:
        """Parse LLM output and inject source metadata via Python post-processing.

        This fixes the source bug: LLM should NOT fill source fields.
        Python fills them from Signal Engine data.

        Args:
            strict: If False (default), strip invalid enum values instead of crashing.
                    Invalid values are logged as warnings.
        """
        import logging
        _logger = logging.getLogger(__name__)

        # Strip source from LLM output if present (LLM shouldn't set it)
        raw.pop("source", None)

        if strict:
            vector = cls.model_validate(raw)
        else:
            # Lenient mode: strip invalid enum values from list fields
            allowed = get_allowed_labels()
            _sanitize_list(raw, ["analysis", "vocal", "vocal_texture"], allowed["vocal_texture"], _logger)
            _sanitize_list(raw, ["analysis", "genre", "secondary_shading"], allowed["secondary_shading"], _logger)
            _sanitize_list(raw, ["prompt_transfer", "preserve_tags"], allowed["preserve_tags"], _logger)
            _sanitize_list(raw, ["prompt_transfer", "avoid_tags"], allowed["avoid_tags"], _logger)

            vector = cls.model_validate(raw)

        vector.inject_source(audio_file, metrics)
        return vector


# ──────────────────────────────────────────────────────
# Allowed values export (for prompt construction)
# ──────────────────────────────────────────────────────

def _sanitize_list(
    data: dict, path: list[str], allowed: set[str], logger: Any = None,
) -> None:
    """Remove invalid values from a nested list field, logging warnings."""
    obj = data
    for key in path[:-1]:
        obj = obj.get(key, {})
        if not isinstance(obj, dict):
            return
    field = path[-1]
    if field in obj and isinstance(obj[field], list):
        invalid = [v for v in obj[field] if v not in allowed]
        if invalid and logger:
            logger.warning(f"Stripped invalid {'.'.join(path)} values: {invalid}")
        obj[field] = [v for v in obj[field] if v in allowed]


def get_allowed_labels() -> dict[str, set[str]]:
    """Return all allowed label values for prompt template construction."""
    return {
        "vocal_presence": {"none", "low", "medium", "high"},
        "lead_focus": {"vocal_led", "instrument_led", "balanced"},
        "vocal_attack": {"soft", "neutral", "hard"},
        "vocal_position": {"front", "mid", "back"},
        "vocal_texture": {
            "clean", "rounded", "breathy", "airy", "plain",
            "intimate", "gentle", "firm",
        },
        "low_mid_density": {"low", "medium", "high"},
        "mid_focus": {"low", "medium", "high"},
        "top_end_presence": {"low", "medium", "high"},
        "brightness": {"dark", "neutral", "bright"},
        "reverb_tail": {"short", "medium", "long"},
        "stereo_width": {"narrow", "medium", "wide"},
        "pad_spread": {"none", "low", "medium", "high"},
        "percussion_presence": {"none", "low", "medium", "high"},
        "rhythm_driver": {
            "vocal_phrase", "piano_pattern", "drums", "bass", "pad_pulse", "mixed",
        },
        "groove_motion": {"static", "gentle_push", "rolling", "pulsed"},
        "dynamic_shape": {"flat", "gradual_build", "stepwise_build", "peak_and_release"},
        "lift_method": {
            "none", "layer_addition", "density_rise", "space_expansion",
            "drum_entry", "hybrid",
        },
        "main_anchor": {
            "ccm_ballad", "studio_worship", "soft_pop_ballad", "piano_ballad",
            "ambient_worship", "orchestral_ballad", "acoustic_ballad", "indie_ballad",
            "cinematic_ballad", "soft_rnb", "dark_electronic", "pop_dance",
            "rock", "rap_bass", "orchestral", "indie",
        },
        "secondary_shading": {
            "studio_polish", "live_worship_hint", "soft_pop", "ambient_pad",
            "piano_led", "orchestral_lift", "restrained_dynamic", "mid_focused_tone",
            "warm_low_mid", "light_percussion", "gentle_build", "wide_space",
        },
        "preserve_tags": {
            "soft_vocal_attack", "front_vocal", "mid_vocal", "rounded_vocal_tone",
            "breathy_vocal_hint", "low_mid_warmth", "mid_focused_body",
            "controlled_top_end", "airy_top_edge", "long_reverb_tail",
            "wide_stereo_pad", "pad_backfill", "low_percussion_presence",
            "piano_led_timing", "vocal_led_timing", "gradual_layer_build",
            "space_expansion_lift", "restrained_dynamics", "studio_polish",
        },
        "avoid_tags": {
            "hard_drum_led_groove", "aggressive_transient_front", "over_bright_top_end",
            "dry_close_mix", "narrow_stereo_field", "heavy_kick_drive",
            "dense_percussion_grid", "abrupt_energy_jump",
            "overwritten_emotion_words", "lyric_meaning_overfit",
        },
    }
