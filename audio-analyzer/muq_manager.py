"""Singleton manager for MuQ and MuQ-MuLan models.

Thread-safe lazy loading and caching of:
  - MuQ-large-msd-iter: 1024-dim audio embeddings
  - MuQ-MuLan-large: text-audio similarity (prompt fidelity)

Pattern mirrors clap_manager.py for consistency.
"""

import logging
import threading
from typing import Optional
import sys
from pathlib import Path

import numpy as np
import torch
import librosa

# Genre labels from consensus module
try:
    from audio_analyzer.consensus.genre_taxonomy import CLAP_CANDIDATE_LABELS
except ImportError:
    try:
        from .consensus.genre_taxonomy import CLAP_CANDIDATE_LABELS
    except ImportError:
        # Fallback if genre_taxonomy unavailable
        CLAP_CANDIDATE_LABELS = [
            "ambient", "blues", "classical", "country", "disco", "electronic",
            "funk", "hip-hop", "indie", "jazz", "metal", "pop", "reggae", "rock"
        ]

logger = logging.getLogger(__name__)


class MuQModelManager:
    """Thread-safe singleton for MuQ + MuQ-MuLan models.
    
    Lazy loading: models load on first use.
    GPU/CPU auto-detection.
    fp32 only (no half precision to avoid NaN).
    """
    
    def __init__(self):
        self._muq_model = None
        self._mulan_model = None
        self._device = None
        self._lock = threading.Lock()
        self._genre_text_embeddings_cache = {}
        logger.info("MuQModelManager initialized (models not yet loaded)")
    
    def _get_device(self) -> str:
        """Detect and cache compute device (cuda or cpu)."""
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"MuQ device: {self._device}")
        return self._device
    
    def _load_muq(self) -> bool:
        """Lazy load MuQ-large-msd-iter model.
        
        Returns:
            True on success, False on failure.
        """
        if self._muq_model is not None:
            return True
        
        with self._lock:
            # Double-check after acquiring lock
            if self._muq_model is not None:
                return True
            
            try:
                logger.info("Loading MuQ-large-msd-iter...")
                from muq import MuQ
                model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
                model = model.to(self._get_device()).float().eval()
                self._muq_model = model
                logger.info("MuQ-large-msd-iter loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to load MuQ-large-msd-iter: {e}")
                return False
    
    def _load_mulan(self) -> bool:
        """Lazy load MuQ-MuLan-large model.
        
        Returns:
            True on success, False on failure.
        """
        if self._mulan_model is not None:
            return True
        
        with self._lock:
            # Double-check after acquiring lock
            if self._mulan_model is not None:
                return True
            
            try:
                logger.info("Loading MuQ-MuLan-large...")
                from muq import MuQMuLan
                model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
                model = model.to(self._get_device()).float().eval()
                self._mulan_model = model
                logger.info("MuQ-MuLan-large loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to load MuQ-MuLan-large: {e}")
                return False
    
    def get_embedding(
        self,
        audio_array: np.ndarray,
        sr: int = 24000
    ) -> Optional[np.ndarray]:
        """Extract 1024-dimensional embedding from audio.
        
        Args:
            audio_array: Mono audio as numpy array (float32, range ~[-1, 1])
            sr: Sample rate of audio_array (will resample to 24kHz if different)
        
        Returns:
            1024-dim numpy array (mean-pooled across time), or None on error.
        """
        if not self._load_muq():
            return None
        
        try:
            # Resample to 24kHz if needed
            if sr != 24000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=24000)
            
            # Convert to torch tensor (fp32)
            audio_tensor = torch.tensor(
                audio_array, dtype=torch.float32, device=self._get_device()
            ).unsqueeze(0)  # (1, samples)
            
            # Run inference
            with torch.no_grad():
                output = self._muq_model(audio_tensor)
            
            # output.last_hidden_state shape: (batch=1, frames, 1024)
            embedding = output.last_hidden_state[0]  # (frames, 1024)
            
            # Mean-pool across time → (1024,)
            embedding_pooled = embedding.mean(dim=0)
            
            return embedding_pooled.cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error computing MuQ embedding: {e}")
            return None
    
    def get_genre_top5(
        self,
        audio_array: np.ndarray,
        sr: int = 24000,
        genre_labels: Optional[list[str]] = None
    ) -> list[tuple[str, float]]:
        """Get top-5 genre predictions using MuQ-MuLan text-audio similarity.
        
        Args:
            audio_array: Mono audio as numpy array
            sr: Sample rate of audio_array
            genre_labels: List of genre names. If None, uses CLAP_CANDIDATE_LABELS.
        
        Returns:
            List of (genre, confidence) tuples, sorted by confidence descending.
            Empty list on error.
        """
        if not self._load_mulan():
            return []
        
        if genre_labels is None:
            genre_labels = CLAP_CANDIDATE_LABELS
        
        try:
            # Resample to 24kHz if needed
            if sr != 24000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=24000)
            
            # Convert to torch tensor (fp32)
            audio_tensor = torch.tensor(
                audio_array, dtype=torch.float32, device=self._get_device()
            ).unsqueeze(0)  # (1, samples)
            
            # Get audio embedding via MuLan
            with torch.no_grad():
                audio_embeds = self._mulan_model(wavs=audio_tensor)
            # audio_embeds shape: (1, embedding_dim)
            
            # Create text prompts for genres
            text_prompts = [f"This is a {genre} song." for genre in genre_labels]
            
            # Get text embeddings (batch all at once for efficiency)
            with torch.no_grad():
                text_embeds = self._mulan_model(texts=text_prompts)
            # text_embeds shape: (num_genres, embedding_dim)
            
            # Compute similarity matrix
            with torch.no_grad():
                similarity_matrix = self._mulan_model.calc_similarity(
                    audio_embeds, text_embeds
                )
            # similarity_matrix shape: (1, num_genres)
            
            similarities = similarity_matrix[0].cpu().numpy()  # (num_genres,)
            
            # Pair genres with scores and sort by score descending
            genre_scores = list(zip(genre_labels, similarities.tolist()))
            genre_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 5
            return genre_scores[:5]
        
        except Exception as e:
            logger.error(f"Error computing genre predictions: {e}")
            return []
    
    def get_prompt_fidelity(
        self,
        audio_array: np.ndarray,
        prompt_text: str,
        sr: int = 24000
    ) -> Optional[float]:
        """Compute prompt↔audio fidelity score using MuQ-MuLan.
        
        Args:
            audio_array: Mono audio as numpy array
            prompt_text: Text prompt to compare
            sr: Sample rate of audio_array
        
        Returns:
            Cosine similarity score (0-1), or None on error.
            Higher = better alignment between prompt and audio.
        """
        if not self._load_mulan():
            return None
        
        try:
            # Resample to 24kHz if needed
            if sr != 24000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=24000)
            
            # Convert to torch tensor (fp32)
            audio_tensor = torch.tensor(
                audio_array, dtype=torch.float32, device=self._get_device()
            ).unsqueeze(0)  # (1, samples)
            
            # Get embeddings
            with torch.no_grad():
                audio_embeds = self._mulan_model(wavs=audio_tensor)
                text_embeds = self._mulan_model(texts=[prompt_text])
            
            # Compute similarity
            with torch.no_grad():
                similarity_matrix = self._mulan_model.calc_similarity(
                    audio_embeds, text_embeds
                )
            
            # similarity_matrix shape: (1, 1)
            fidelity_score = similarity_matrix[0, 0].item()
            
            return fidelity_score
        
        except Exception as e:
            logger.error(f"Error computing prompt fidelity: {e}")
            return None


# Module-level singleton instance
_muq_manager: Optional[MuQModelManager] = None


def get_muq_manager() -> MuQModelManager:
    """Get or create the global MuQ manager singleton.
    
    Returns:
        MuQModelManager instance (thread-safe).
    """
    global _muq_manager
    if _muq_manager is None:
        _muq_manager = MuQModelManager()
    return _muq_manager
