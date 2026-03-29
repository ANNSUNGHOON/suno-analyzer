"""Essentia-based audio feature extraction module.

Extracts key, scale, key strength, danceability, and energy.
Designed as optional dependency — gracefully falls back on import failure.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from .schemas import EssentiaFeatures
except ImportError:
    from schemas import EssentiaFeatures


def analyze(audio_path: str) -> EssentiaFeatures:
    """Extract Essentia features from an audio file.
    
    Loads audio at 44100Hz mono float32 (Essentia standard) and extracts:
    - Key via essentia.standard.KeyExtractor
    - Scale (minor/major)
    - Key strength (confidence 0-1)
    - Danceability (0-1)
    - Energy (normalized 0-1)
    
    Returns EssentiaFeatures object with all extracted values.
    On error (including Essentia not installed), returns EssentiaFeatures
    with defaults and error field set.
    
    Args:
        audio_path: Path to MP3 or WAV file
        
    Returns:
        EssentiaFeatures dataclass instance
    """
    try:
        import essentia.standard as es
    except ImportError:
        logger.warning("Essentia not installed")
        return EssentiaFeatures(error="essentia not installed")
    
    try:
        audio = es.MonoLoader(filename=audio_path, sampleRate=44100)()
    except Exception as e:
        logger.error(f"Failed to load audio with Essentia: {e}")
        return EssentiaFeatures(error=f"load failed: {str(e)}")
    
    result = EssentiaFeatures()
    
    # Key extraction
    try:
        key_ext = es.KeyExtractor()
        key, scale, strength = key_ext(audio)
        result.key = str(key) if key else ""
        result.scale = str(scale) if scale else ""
        result.key_strength = round(float(strength), 4)
    except Exception as e:
        logger.warning(f"Key extraction failed: {e}")
    
    # Danceability
    try:
        dance = es.Danceability()(audio)
        result.danceability = round(float(dance[0]), 4)
    except Exception as e:
        logger.warning(f"Danceability extraction failed: {e}")
    
    # Energy (normalize by dividing by a reasonable max value)
    try:
        energy = es.Energy()(audio)
        # Essentia Energy returns raw energy; normalize to 0-1
        # Typical max observed ~1e6, use that as denominator
        energy_norm = min(float(energy) / 1e6, 1.0)
        result.energy = round(energy_norm, 4)
    except Exception as e:
        logger.warning(f"Energy extraction failed: {e}")
    
    return result


def analyze_from_array(audio_array: np.ndarray, sr: int = 44100) -> EssentiaFeatures:
    """Extract Essentia features from a pre-loaded audio array.
    
    Essentia expects 44100Hz mono float32. If your array is in a different
    format, you must resample/convert before passing here.
    
    Args:
        audio_array: Audio time series (numpy array, float32, mono)
        sr: Sample rate (default 44100, should match array)
        
    Returns:
        EssentiaFeatures dataclass instance
    """
    try:
        import essentia.standard as es
    except ImportError:
        logger.warning("Essentia not installed")
        return EssentiaFeatures(error="essentia not installed")
    
    try:
        # Ensure audio is float32
        audio_array = np.asarray(audio_array, dtype=np.float32)
    except Exception as e:
        logger.error(f"Failed to convert audio array to float32: {e}")
        return EssentiaFeatures(error=f"array conversion failed: {str(e)}")
    
    result = EssentiaFeatures()
    
    # Key extraction
    try:
        key_ext = es.KeyExtractor()
        key, scale, strength = key_ext(audio_array)
        result.key = str(key) if key else ""
        result.scale = str(scale) if scale else ""
        result.key_strength = round(float(strength), 4)
    except Exception as e:
        logger.warning(f"Key extraction failed: {e}")
    
    # Danceability
    try:
        dance = es.Danceability()(audio_array)
        result.danceability = round(float(dance[0]), 4)
    except Exception as e:
        logger.warning(f"Danceability extraction failed: {e}")
    
    # Energy
    try:
        energy = es.Energy()(audio_array)
        energy_norm = min(float(energy) / 1e6, 1.0)
        result.energy = round(energy_norm, 4)
    except Exception as e:
        logger.warning(f"Energy extraction failed: {e}")
    
    return result
