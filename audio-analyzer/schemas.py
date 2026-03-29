"""Pipeline v3 data schemas — Pydantic v2 models for Signal Engine v2.

5 core schemas for the Suno language reverse-engineering closed loop:
  - TrackAnalysisV2: Signal Engine output (Essentia + Librosa + MuQ)
  - MusicInference: Prism's LLM reasoning on TrackAnalysis
  - SunoPrompt: Muse's reverse prompt output
  - ComparisonResult: A↔B comparison analysis
  - SunoLanguageModel: Forge's aggregated keyword effect map

All models support to_dict() / from_dict() for JSON serialization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────────────
# Sub-feature models
# ──────────────────────────────────────────────────────

class LibrosaFeatures(BaseModel):
    """Librosa-extracted audio features."""
    model_config = ConfigDict(frozen=False)

    bpm: float = 0.0
    key: str = ""                       # pitch class (C, C#, D, ...)
    duration_seconds: float = 0.0
    energy: float = 0.0                 # RMS normalized 0-1
    spectral_centroid: float = 0.0      # Hz
    spectral_bandwidth: float = 0.0     # Hz
    spectral_rolloff: float = 0.0       # Hz
    dynamic_range_db: float = 0.0       # dB
    zero_crossing_rate: float = 0.0     # 0-1
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = self.model_dump(exclude_none=True)
        if "error" in d and d["error"] is None:
            del d["error"]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> LibrosaFeatures:
        return cls.model_validate(data)


class EssentiaFeatures(BaseModel):
    """Essentia-extracted audio features."""
    model_config = ConfigDict(frozen=False)

    key: str = ""
    scale: str = ""                     # major / minor
    key_strength: float = 0.0           # 0-1
    danceability: float = 0.0           # 0-1
    energy: float = 0.0                 # 0-1 normalized
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = self.model_dump(exclude_none=True)
        if "error" in d and d["error"] is None:
            del d["error"]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> EssentiaFeatures:
        return cls.model_validate(data)


class MuQFeatures(BaseModel):
    """MuQ embedding + MuQ-MuLan genre/fidelity features."""
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    embedding: list[float] = Field(default_factory=list)            # 1024-dim
    genre_top5: list[tuple[str, float]] = Field(default_factory=list)
    prompt_fidelity: Optional[float] = None                         # MuQ-MuLan score
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = self.model_dump(exclude_none=True)
        # Don't serialize the full 1024-dim embedding by default
        if "embedding" in d and len(d["embedding"]) > 0:
            d["embedding_dim"] = len(d["embedding"])
            del d["embedding"]
        return d

    def to_dict_full(self) -> dict:
        """Include full embedding vector (for ML / storage)."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict) -> MuQFeatures:
        return cls.model_validate(data)


class CrossValidation(BaseModel):
    """Librosa ↔ Essentia cross-validation results."""
    model_config = ConfigDict(frozen=False)

    bpm_match: bool = True
    bpm_librosa: float = 0.0
    bpm_essentia: float = 0.0
    key_match: bool = True
    key_librosa: str = ""
    key_essentia: str = ""
    energy_match: bool = True
    energy_librosa: float = 0.0
    energy_essentia: float = 0.0
    matched_features: list[str] = Field(default_factory=list)
    mismatched_features: list[str] = Field(default_factory=list)
    agreement_ratio: float = 1.0
    confidence_penalty: float = 0.0

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> CrossValidation:
        return cls.model_validate(data)


# ──────────────────────────────────────────────────────
# Core pipeline schemas
# ──────────────────────────────────────────────────────

class TrackAnalysisV2(BaseModel):
    """Signal Engine v2 unified output.

    Produced by: Prism (execution via SignalEngine)
    Consumed by: Prism (inference), Forge (ML training)
    """
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    track_id: str
    audio_path: str = ""
    librosa: LibrosaFeatures = Field(default_factory=LibrosaFeatures)
    essentia: EssentiaFeatures = Field(default_factory=EssentiaFeatures)
    muq: MuQFeatures = Field(default_factory=MuQFeatures)
    cross_validation: CrossValidation = Field(default_factory=CrossValidation)
    processing_ms: float = 0.0
    engine_version: str = "2.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Convenience properties
    @property
    def bpm(self) -> float:
        return self.librosa.bpm

    @property
    def key(self) -> str:
        return self.librosa.key or self.essentia.key

    @property
    def scale(self) -> str:
        return self.essentia.scale

    def to_dict(self) -> dict:
        """Serialize without embedding (for logging/API)."""
        return {
            "track_id": self.track_id,
            "audio_path": self.audio_path,
            "librosa": self.librosa.to_dict(),
            "essentia": self.essentia.to_dict(),
            "muq": self.muq.to_dict(),
            "cross_validation": self.cross_validation.to_dict(),
            "processing_ms": self.processing_ms,
            "engine_version": self.engine_version,
            "created_at": self.created_at,
        }

    def to_dict_full(self) -> dict:
        """Serialize with full embedding (for DB storage)."""
        d = self.to_dict()
        d["muq"] = self.muq.to_dict_full()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> TrackAnalysisV2:
        return cls.model_validate(data)


class MusicInference(BaseModel):
    """Prism's music-aware reasoning on a TrackAnalysis.

    Produced by: Prism (LLM inference)
    Consumed by: Muse (prompt generation), Forge (feedback)
    """
    model_config = ConfigDict(frozen=False)

    track_id: str
    genre_primary: str = ""
    genre_alternatives: list[str] = Field(default_factory=list)
    mood: str = ""
    instruments: list[str] = Field(default_factory=list)
    vocal_type: str = ""                # clean, raspy, falsetto, none, etc.
    production_style: str = ""
    energy_level: str = ""              # low, medium, high
    confidence: float = 0.0             # 0-10
    reasoning: str = ""
    source_analysis_id: str = ""        # → TrackAnalysisV2.track_id
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> MusicInference:
        return cls.model_validate(data)


class SunoPrompt(BaseModel):
    """Muse's reverse prompt for Suno generation.

    Produced by: Muse
    Consumed by: 성훈 (manual Suno generation)
    """
    model_config = ConfigDict(frozen=False)

    track_id: str                       # source track being reverse-prompted
    foundation: str = ""                # Foundation layer (DB B-Layer tokens + BPM)
    performance: str = ""               # Performance layer (550-750 char English prose)
    genre_tags: list[str] = Field(default_factory=list)
    source_inference_id: str = ""       # → MusicInference.track_id
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> SunoPrompt:
        return cls.model_validate(data)


class ComparisonResult(BaseModel):
    """A ↔ B comparison analysis.

    Produced by: Prism (comparison mode)
    Consumed by: Forge (SunoLanguageModel aggregation), Conductor
    """
    model_config = ConfigDict(frozen=False)

    track_a_id: str                     # original
    track_b_id: str                     # Suno-generated
    prompt_used: str = ""               # the SunoPrompt that generated B

    # Comparison metrics (priority order)
    muq_embedding_cosine: float = 0.0   # 1. semantic similarity
    muq_genre_rank_shift: dict = Field(default_factory=dict)   # 2. {genre: rank_delta}
    spectral_centroid_delta: float = 0.0   # 3. timbre response
    energy_delta: float = 0.0           # 4. energy level change
    prompt_fidelity_delta: float = 0.0  # 5. MuQ-MuLan fidelity change
    bpm_delta: float = 0.0             # 6a. BPM change
    key_delta: str = ""                # 6b. key change description

    # Keyword effect inference
    keyword_effects: dict = Field(default_factory=dict)  # {keyword: {metric: delta}}

    overall_similarity: float = 0.0     # weighted composite 0-1
    notes: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> ComparisonResult:
        return cls.model_validate(data)


class SunoLanguageModel(BaseModel):
    """Aggregated Suno language understanding model.

    Produced by: Forge (aggregation over ComparisonResults)
    Consumed by: Muse (prompt optimization), Genre DB (updates)
    """
    model_config = ConfigDict(frozen=False)

    version: str = "1.0"
    total_comparisons: int = 0
    keyword_effect_map: dict = Field(default_factory=dict)
    # {keyword: {metric: {mean_delta: float, std: float, sample_count: int}}}
    genre_response_map: dict = Field(default_factory=dict)
    # {genre: {keyword: effectiveness_score}}
    convergence_metrics: dict = Field(default_factory=dict)
    # {new_keyword_rate: float, avg_cosine: float}
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> SunoLanguageModel:
        return cls.model_validate(data)
