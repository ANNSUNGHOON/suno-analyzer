"""Librosa-based audio feature extraction module.

Extracts tempo, key, spectral features, energy, and dynamic range.
All features are extracted with individual error handling for robustness.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from .schemas import LibrosaFeatures
except ImportError:
    from schemas import LibrosaFeatures

# Key mapping: chroma index → pitch class
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def analyze(audio_path: str) -> LibrosaFeatures:
    """Extract Librosa features from an audio file.

    Loads audio at sr=22050Hz (Librosa default) and delegates to _extract().

    Args:
        audio_path: Path to MP3 or WAV file.

    Returns:
        LibrosaFeatures with all extracted values (defaults on error).
    """
    try:
        import librosa
    except ImportError:
        logger.error("librosa not installed")
        return LibrosaFeatures(error="librosa not installed")

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return LibrosaFeatures(error=f"load failed: {str(e)}")

    return _extract(y, sr)


def analyze_from_array(y: np.ndarray, sr: int = 22050) -> LibrosaFeatures:
    """Extract Librosa features from a pre-loaded audio array.

    Args:
        y: Audio time series (numpy array, mono).
        sr: Sample rate (default 22050).

    Returns:
        LibrosaFeatures dataclass instance.
    """
    try:
        import librosa  # noqa: F401 — ensure librosa is available
    except ImportError:
        logger.error("librosa not installed")
        return LibrosaFeatures(error="librosa not installed")

    return _extract(y, sr)


def _extract(y: np.ndarray, sr: int) -> LibrosaFeatures:
    """Core extraction logic. Called by both analyze() and analyze_from_array().

    Each feature is extracted in its own try/except block for robustness.
    """
    import librosa

    result = LibrosaFeatures()

    # Duration
    try:
        result.duration_seconds = round(float(librosa.get_duration(y=y, sr=sr)), 1)
    except Exception as e:
        logger.warning(f"Duration extraction failed: {e}")

    # BPM
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        result.bpm = round(bpm, 1)
    except Exception as e:
        logger.warning(f"BPM extraction failed: {e}")

    # Key (chroma_cqt → argmax → pitch class)
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_idx = int(np.argmax(chroma_mean))
        result.key = _PITCH_CLASSES[key_idx]
    except Exception as e:
        logger.warning(f"Key extraction failed: {e}")

    # RMS — compute once, reuse for energy + dynamic range
    rms = None
    try:
        rms = librosa.feature.rms(y=y)
    except Exception as e:
        logger.warning(f"RMS extraction failed: {e}")

    # Energy (RMS mean, normalized to 0-1)
    if rms is not None:
        try:
            energy_raw = float(np.mean(rms))
            result.energy = round(min(energy_raw * 5, 1.0), 4)
        except Exception as e:
            logger.warning(f"Energy normalization failed: {e}")

    # Dynamic range (max_db - min_db of RMS)
    if rms is not None:
        try:
            rms_db = librosa.amplitude_to_db(rms)
            dynamic_range = float(np.max(rms_db) - np.min(rms_db))
            result.dynamic_range_db = round(dynamic_range, 1)
        except Exception as e:
            logger.warning(f"Dynamic range extraction failed: {e}")

    # Spectral centroid
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        result.spectral_centroid = round(float(np.mean(cent)), 1)
    except Exception as e:
        logger.warning(f"Spectral centroid extraction failed: {e}")

    # Spectral bandwidth
    try:
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        result.spectral_bandwidth = round(float(np.mean(bw)), 1)
    except Exception as e:
        logger.warning(f"Spectral bandwidth extraction failed: {e}")

    # Spectral rolloff
    try:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        result.spectral_rolloff = round(float(np.mean(rolloff)), 1)
    except Exception as e:
        logger.warning(f"Spectral rolloff extraction failed: {e}")

    # Zero-crossing rate
    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        result.zero_crossing_rate = round(float(np.mean(zcr)), 4)
    except Exception as e:
        logger.warning(f"Zero-crossing rate extraction failed: {e}")

    return result
