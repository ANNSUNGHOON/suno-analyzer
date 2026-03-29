"""Audio Analyzer — Signal Engine v2 for Music_Jarvis Pipeline v3.

Usage:
    from audio_analyzer import SignalEngine, TrackAnalysisV2

    engine = SignalEngine()
    result = engine.analyze("track.wav")
"""

from .signal_engine import SignalEngine
from .schemas import (
    TrackAnalysisV2,
    LibrosaFeatures,
    EssentiaFeatures,
    MuQFeatures,
    CrossValidation,
    MusicInference,
    SunoPrompt,
    ComparisonResult,
    SunoLanguageModel,
)

__all__ = [
    "SignalEngine",
    "TrackAnalysisV2",
    "LibrosaFeatures",
    "EssentiaFeatures",
    "MuQFeatures",
    "CrossValidation",
    "MusicInference",
    "SunoPrompt",
    "ComparisonResult",
    "SunoLanguageModel",
]

__version__ = "2.0.0"
