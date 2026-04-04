"""Signal Engine v2 – Unified audio analysis interface.

Orchestrates three analysis backends:
  - Librosa: BPM, key, spectral features, energy, dynamics
  - Essentia: key, scale, danceability, energy (cross-validation)
  - MuQ: 1024-dim embedding, genre top5 (MuQ-MuLan), prompt fidelity

Produced by: Prism (execution)
Consumed by: Prism (inference), Forge (ML)

Usage:
    from audio_analyzer.signal_engine import SignalEngine

    engine = SignalEngine()
    result = engine.analyze("path/to/audio.wav")
    result = engine.analyze("path/to/audio.wav", prompt_text="electronic techno driving")

    # Batch
    results = engine.analyze_batch(["track1.wav", "track2.wav"])
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 🛠  Internal imports (graceful fallback for both package and script modes) 🛠

try:
    from . import librosa_analyzer
    from . import essentia_analyzer
    from .muq_manager import get_muq_manager
    from .schemas import (
        TrackAnalysisV2,
        LibrosaFeatures,
        EssentiaFeatures,
        MuQFeatures,
        CrossValidation,
    )
    from .prism_schema import PrismVector, ObjectiveMetrics
    from .audio_perception import create_client as create_perception_client
except ImportError:
    import librosa_analyzer  # type: ignore
    import essentia_analyzer  # type: ignore
    from muq_manager import get_muq_manager  # type: ignore
    from schemas import (  # type: ignore
        TrackAnalysisV2,
        LibrosaFeatures,
        EssentiaFeatures,
        MuQFeatures,
        CrossValidation,
    )
    from prism_schema import PrismVector, ObjectiveMetrics  # type: ignore
    from audio_perception import create_client as create_perception_client  # type: ignore

# 🛠  Cross-validation thresholds (from existing stage1_signal.py) 🛠

XVAL_BPM_TOLERANCE = 5.0
XVAL_ENERGY_TOLERANCE = 0.15
XVAL_PENALTY_PER_MISMATCH = 0.1


class SignalEngine:
    """Signal Engine v2 – unified analysis orchestrator.

    Thread-safe. MuQ models loaded lazily on first use.
    Essentia is optional (graceful degradation if not installed).
    """

    VERSION = "2.0"

    def __init__(
        self,
        enable_muq: bool = True,
        enable_essentia: bool = True,
        enable_audio_perception: bool = False,
        perception_provider: str = "gemini",
        genre_labels: Optional[list[str]] = None,
    ) -> None:
        """
        Args:
            enable_muq: Enable MuQ embedding extraction (GPU recommended).
            enable_essentia: Enable Essentia analysis (optional dependency).
            enable_audio_perception: Enable Layer 2 Audio Perception (API LLM).
            perception_provider: "gemini" (default) or "qwen".
            genre_labels: Custom genre labels for MuQ-MuLan.
                          Defaults to CLAP_CANDIDATE_LABELS from genre_taxonomy.
        """
        self._enable_muq = enable_muq
        self._enable_essentia = enable_essentia
        self._enable_audio_perception = enable_audio_perception
        self._perception_provider = perception_provider
        self._perception_client = None
        self._genre_labels = genre_labels

        # Try loading default genre labels from genre_taxonomy
        if self._genre_labels is None:
            try:
                parent = str(Path(__file__).resolve().parent.parent)
                if parent not in sys.path:
                    sys.path.insert(0, parent)
                from genre_taxonomy import CLAP_CANDIDATE_LABELS  # type: ignore
                self._genre_labels = list(CLAP_CANDIDATE_LABELS)
                logger.info(
                    f"Loaded {len(self._genre_labels)} genre labels from genre_taxonomy"
                )
            except ImportError:
                logger.warning(
                    "genre_taxonomy not found; using default genre labels"
                )
                self._genre_labels = [
                    "electronic", "techno", "house", "trance", "ambient",
                    "rock", "alternative rock", "indie rock", "hard rock",
                    "metal", "heavy metal", "pop", "synth-pop", "electropop",
                    "indie pop", "hip-hop", "trap", "lo-fi hip-hop",
                    "jazz", "smooth jazz", "fusion", "classical", "orchestral",
                    "folk", "country", "blues", "r&b", "soul", "funk", "disco",
                    "reggae", "latin", "world music", "worship", "gospel",
                ]

    # 🛠  Public API 🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠

    def analyze(
        self,
        audio_path: str,
        track_id: Optional[str] = None,
        prompt_text: Optional[str] = None,
    ) -> TrackAnalysisV2:
        """Analyze a single audio file with all available backends.

        Args:
            audio_path: Path to audio file (wav, mp3, flac, etc.)
            track_id: Optional identifier. Defaults to filename stem.
            prompt_text: Optional text for MuQ-MuLan prompt fidelity scoring.

        Returns:
            TrackAnalysisV2 with all available features populated.
        """
        start_time = time.perf_counter()

        if track_id is None:
            track_id = Path(audio_path).stem

        result = TrackAnalysisV2(track_id=track_id, audio_path=str(audio_path))

        # 1) Librosa – always available, primary analyzer
        try:
            result.librosa = librosa_analyzer.analyze(audio_path)
            logger.debug(f"Librosa OK: bpm={result.librosa.bpm}, key={result.librosa.key}")
        except Exception as e:
            logger.error(f"Librosa analysis failed for {track_id}: {e}")
            result.librosa = LibrosaFeatures(error=str(e))

        # 2) Essentia – optional
        if self._enable_essentia:
            try:
                result.essentia = essentia_analyzer.analyze(audio_path)
                logger.debug(
                    f"Essentia OK: key={result.essentia.key}, scale={result.essentia.scale}"
                )
            except Exception as e:
                logger.warning(f"Essentia analysis failed for {track_id}: {e}")
                result.essentia = EssentiaFeatures(error=str(e))

        # 3) Cross-validate Librosa ↔ Essentia
        result.cross_validation = self._cross_validate(result.librosa, result.essentia)

        # 4) MuQ – optional, GPU recommended
        if self._enable_muq:
            try:
                result.muq = self._run_muq(audio_path, prompt_text)
                logger.debug(
                    f"MuQ OK: emb_dim={len(result.muq.embedding)}, "
                    f"genres={len(result.muq.genre_top5)}"
                )
            except Exception as e:
                logger.warning(f"MuQ analysis failed for {track_id}: {e}")
                result.muq = MuQFeatures(error=str(e))

        # 5) Audio Perception Layer (Layer 2) – optional API LLM
        if self._enable_audio_perception:
            try:
                prism_vector = self._run_audio_perception(audio_path, result)
                result.prism_vector = prism_vector.to_dict()
                logger.debug(
                    f"Audio Perception OK: anchor={prism_vector.analysis.genre.main_anchor}"
                )
            except Exception as e:
                logger.warning(f"Audio Perception failed for {track_id}: {e}")
                # Non-fatal: L2 failure doesn't break L1 results

        result.processing_ms = round((time.perf_counter() - start_time) * 1000, 1)
        result.engine_version = self.VERSION

        logger.info(
            f"Signal Engine v{self.VERSION} complete: {track_id} "
            f"({result.processing_ms:.0f}ms)"
        )
        return result

    def analyze_batch(
        self,
        audio_paths: list[str],
        track_ids: Optional[list[str]] = None,
        prompt_texts: Optional[list[str]] = None,
    ) -> list[TrackAnalysisV2]:
        """Analyze multiple audio files sequentially.

        MuQ is GPU-bound so parallel execution gives no benefit.
        For CPU-only mode, consider using multiprocessing externally.

        Args:
            audio_paths: List of audio file paths.
            track_ids: Optional list of track identifiers.
            prompt_texts: Optional list of prompt texts for fidelity scoring.

        Returns:
            List of TrackAnalysisV2 results.
        """
        n = len(audio_paths)
        if track_ids is None:
            track_ids = [None] * n  # type: ignore
        if prompt_texts is None:
            prompt_texts = [None] * n  # type: ignore

        results = []
        for i, (path, tid, prompt) in enumerate(
            zip(audio_paths, track_ids, prompt_texts)
        ):
            logger.info(f"Batch [{i + 1}/{n}]: {path}")
            results.append(self.analyze(path, track_id=tid, prompt_text=prompt))
        return results

    # 🛠  Audio Perception (Layer 2)

    def _run_audio_perception(
        self, audio_path: str, analysis: TrackAnalysisV2
    ) -> "PrismVector":
        """Run Layer 2 Audio Perception — extract closed-label vector via API LLM."""
        if self._perception_client is None:
            self._perception_client = create_perception_client(self._perception_provider)
            logger.info(f"Audio Perception client created: {self._perception_provider}")

        # Build ObjectiveMetrics from L1 results
        metrics = ObjectiveMetrics(
            key=analysis.librosa.key or analysis.essentia.key,
            scale=analysis.essentia.scale,
            key_strength=analysis.essentia.key_strength,
            bpm=analysis.librosa.bpm,
            spectral_centroid=analysis.librosa.spectral_centroid,
            energy_normalized=analysis.librosa.energy,
        )

        return self._perception_client.extract(audio_path, metrics)

    # 🛠  MuQ Analysis 🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠

    def _run_muq(
        self, audio_path: str, prompt_text: Optional[str] = None
    ) -> MuQFeatures:
        """Run MuQ + MuQ-MuLan analysis on an audio file."""
        import librosa as lr

        muq_mgr = get_muq_manager()
        features = MuQFeatures()

        # Load at 24kHz for MuQ
        y, sr = lr.load(audio_path, sr=24000, mono=True)

        # Embedding (1024-dim, mean-pooled)
        embedding = muq_mgr.get_embedding(y, sr=24000)
        if embedding is not None:
            features.embedding = embedding.tolist()

        # Genre top5 via MuQ-MuLan
        if self._genre_labels:
            genre_top5 = muq_mgr.get_genre_top5(
                y, sr=24000, genre_labels=self._genre_labels
            )
            if genre_top5:
                features.genre_top5 = genre_top5

        # Prompt fidelity (only if prompt provided)
        if prompt_text:
            fidelity = muq_mgr.get_prompt_fidelity(y, prompt_text, sr=24000)
            if fidelity is not None:
                features.prompt_fidelity = fidelity

        return features

    # 🛠  Cross-Validation 🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠🛠

    def _cross_validate(
        self, librosa_feat: LibrosaFeatures, essentia_feat: EssentiaFeatures
    ) -> CrossValidation:
        """Cross-validate Librosa ↔ Essentia overlapping features."""
        xval = CrossValidation()

        xval.bpm_librosa = librosa_feat.bpm
        xval.key_librosa = librosa_feat.key
        xval.energy_librosa = librosa_feat.energy
        xval.key_essentia = essentia_feat.key
        xval.energy_essentia = essentia_feat.energy

        matched: list[str] = []
        mismatched: list[str] = []

        # BPM – Essentia doesn't extract BPM in current setup, skip if 0
        if xval.bpm_essentia > 0:
            xval.bpm_match = abs(xval.bpm_librosa - xval.bpm_essentia) <= XVAL_BPM_TOLERANCE
            (matched if xval.bpm_match else mismatched).append("bpm")

        # Key
        if xval.key_librosa and xval.key_essentia:
            xval.key_match = xval.key_librosa.upper() == xval.key_essentia.upper()
            (matched if xval.key_match else mismatched).append("key")

        # Energy
        if xval.energy_librosa > 0 or xval.energy_essentia > 0:
            xval.energy_match = (
                abs(xval.energy_librosa - xval.energy_essentia) <= XVAL_ENERGY_TOLERANCE
            )
            (matched if xval.energy_match else mismatched).append("energy")

        xval.matched_features = matched
        xval.mismatched_features = mismatched

        total = len(matched) + len(mismatched)
        xval.agreement_ratio = len(matched) / total if total > 0 else 1.0
        xval.confidence_penalty = len(mismatched) * XVAL_PENALTY_PER_MISMATCH

        return xval
