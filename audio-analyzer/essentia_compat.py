"""Essentia-compatible feature extraction using only librosa + numpy.
Provides scale (major/minor), key_strength, danceability.
Uses Krumhansl-Kessler key profiles and onset-based danceability.
"""
import numpy as np
import librosa

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def detect_key_and_scale(y, sr=22050):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    best_corr = -1
    best_key = "C"
    best_scale = "major"
    for shift in range(12):
        rotated = np.roll(chroma_mean, -shift)
        major_corr = np.corrcoef(rotated, MAJOR_PROFILE)[0, 1]
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = pitch_classes[shift]
            best_scale = "major"
        minor_corr = np.corrcoef(rotated, MINOR_PROFILE)[0, 1]
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = pitch_classes[shift]
            best_scale = "minor"
    key_strength = max(0.0, min(1.0, float(best_corr)))
    return best_key, best_scale, round(key_strength, 4)

def estimate_danceability(y, sr=22050):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if len(onset_env) < 10:
        return 0.0
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
    ac = ac / (ac[0] + 1e-8)
    hop_length = 512
    beat_period_frames = int(60.0 / max(tempo_val, 30) * sr / hop_length)
    beat_ac = float(ac[beat_period_frames]) if beat_period_frames < len(ac) else 0.0
    if len(beats) >= 3:
        ibi = np.diff(beats).astype(float)
        ibi_cv = float(np.std(ibi) / (np.mean(ibi) + 1e-8))
        regularity = max(0.0, 1.0 - ibi_cv)
    else:
        regularity = 0.0
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    duration = len(y) / sr
    onset_density = len(onset_frames) / max(duration, 1.0)
    onset_factor = min(onset_density / 4.0, 1.0)
    danceability = (beat_ac * 1.2 + regularity * 1.0 + onset_factor * 0.8)
    return round(float(max(0.0, danceability)), 4)

def extract_essentia_compat(y, sr=22050):
    key, scale, key_strength = detect_key_and_scale(y, sr)
    danceability = estimate_danceability(y, sr)
    return {
        "essentia_compat_key": key,
        "essentia_compat_scale": scale,
        "essentia_compat_key_strength": key_strength,
        "essentia_compat_danceability": danceability,
    }