"""
SAE Vector Registry

Manages retrieval and caching of Sparse Autoencoder feature vectors.
Uses sae_lens to load and query SAE features.
"""

import torch
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger("sae_registry")


@dataclass
class SAERelease:
    """Metadata about an SAE release."""
    release_id: str
    model_name: str
    sae_type: str
    n_features: int
    layer: int


@dataclass
class DistilledSAEConfig:
    """Configuration for a distilled SAE (lightweight soul vectors)."""
    model_id: str
    path: str
    layer: int
    d_model: int
    n_features: int
    source_set: str
    description: str


class SAEVectorRegistry:
    """
    Registry for managing SAE feature vectors.
    
    This class handles loading SAEs from sae_lens and providing
    feature vectors for steering operations.
    
    Supports:
    - Gemma Scope SAEs
    - Custom SAE loading
    - Distilled SAEs (lightweight soul vectors from .pt files)
    - Vector caching for performance
    """
    
    # Registry of available distilled SAEs
    DISTILLED_SAE_CONFIGS: Dict[str, DistilledSAEConfig] = {
        "llama-3.1-8b-soul": DistilledSAEConfig(
            model_id="llama3.1-8b-it",
            path="data/llama_8b_soul_vectors.pt",
            layer=19,
            d_model=4096,
            n_features=0,  # Will be determined at load time
            source_set="resid-post-aa",
            description="Llama 3.1 8B distilled soul vectors (Goodfire SAE layer 19)",
        ),
        "llama-3.1-8b-eleuther": DistilledSAEConfig(
            model_id="llama3.1-8b-it",
            path="data/llama_8b_eleuther.pt",
            layer=20,
            d_model=4096,
            n_features=131072,  # EleutherAI 32x expansion
            source_set="resid-post-aa",
            description="Llama 3.1 8B distilled soul vectors (EleutherAI SAE layer 20, 32x)",
        ),
    }
    
    def __init__(
        self,
        release: str = "gemma-scope-2b-pt-res-jh",
        sae_type: str = "jump_relu",
        device: str = "cpu",
        sae=None,  # Pre-loaded SAE object (from sae_lens SAE.from_pretrained)
        distilled_path: Optional[str] = None,  # Path to distilled .pt file
    ):
        """
        Initialize the SAE registry.

        Args:
            release: SAE release ID (used for metadata / auto-load)
            sae_type: Type of SAE (jump_relu, standard, etc.)
            device: Device to load SAE on (used when auto-loading)
            sae: Pre-loaded SAE object. If provided, skips auto-load via .load().
                 This is the primary path when build_soul_vectors.py loads the SAE
                 externally and passes it in.
            distilled_path: Path to a distilled SAE .pt file (e.g., llama_8b_soul_vectors.pt).
                 If provided, loads lightweight soul vectors instead of full SAE.
        """
        self.release = release
        self.sae_type = sae_type
        self.device = device
        self.distilled_path = distilled_path
        self._distilled_data: Optional[Dict] = None

        # Accept a pre-loaded SAE (preferred path)
        self.sae = sae
        self.model = None

        if distilled_path is not None:
            # Load distilled vectors (lightweight path)
            self._load_distilled(distilled_path)
        elif sae is not None:
            # Derive n_features from the W_dec shape
            self.n_features = sae.W_dec.shape[0]
            logger.info(
                f"SAEVectorRegistry initialized with pre-loaded SAE | "
                f"release={release} | n_features={self.n_features} | "
                f"d_model={sae.W_dec.shape[1]}"
            )
        else:
            self.n_features = 0

        # Cache for feature vectors
        self._vector_cache: Dict[int, torch.Tensor] = {}
        self._cache_enabled = True
    
    def _load_distilled(self, path: str):
        """Load a distilled SAE from a .pt file."""
        try:
            data = torch.load(path, map_location=self.device, weights_only=False)
            self._distilled_data = data
            
            # Extract metadata
            self.n_features = data.get("n_features_soul", 0)
            d_model = data.get("d_model", 0)
            model_id = data.get("model", "unknown")
            layer = data.get("layer", 0)
            
            # Create a lightweight mock sae object
            self.sae = self._create_distilled_sae_mock(data)
            
            logger.info(
                f"SAEVectorRegistry initialized with distilled SAE | "
                f"path={path} | model={model_id} | layer={layer} | "
                f"n_features={self.n_features} | d_model={d_model}"
            )
        except Exception as e:
            logger.error(f"Failed to load distilled SAE from {path}: {e}")
            raise
    
    def _create_distilled_sae_mock(self, data: Dict):
        """Create a mock SAE object from distilled data."""
        class DistilledSAEMock:
            def __init__(self, data):
                d_model = data.get("d_model", 0)
                n_features = data.get("n_features_soul", 0)
                
                self.cfg = type('obj', (object,), {
                    'd_in': d_model,
                    'd_out': d_model,
                    'neuron_dim': n_features,
                    'architecture': 'distilled',
                    'model_name': data.get("model", "unknown"),
                })()
                
                # W_dec_soul tiene shape [d_model, n_soul_features] en formato Goodfire
                # (cada columna es un feature vector)
                self.W_dec = data.get("W_dec_soul", torch.zeros((d_model, n_features)))
                self.W_enc = data.get("W_enc_soul")
                
                # Store v_c and v_d as special vectors
                self.v_c = data.get("v_c")
                self.v_d = data.get("v_d")
                
                # Store feature ID mapping
                self.c_feature_ids = data.get("c_feature_ids", [])
                self.d_feature_ids = data.get("d_feature_ids", [])
                self.all_soul_ids = self.c_feature_ids + self.d_feature_ids
            
            def get_feature_vector(self, feature_id: int) -> Optional[torch.Tensor]:
                """
                Get the decoder vector for a specific feature.
                
                For Goodfire format: W_dec has shape [d_model, n_features]
                where each column is a feature vector.
                
                Args:
                    feature_id: Index into the soul features (0 to n_soul_features-1)
                    
                Returns:
                    Feature vector of shape [d_model], or None if out of range
                """
                if feature_id < 0 or feature_id >= len(self.all_soul_ids):
                    return None
                
                # W_dec_soul is [d_model, n_soul_features], get column
                if self.W_dec.shape[1] > feature_id:
                    return self.W_dec[:, feature_id]
                return None
            
            def get_feature_id_by_index(self, idx: int) -> int:
                """Map a soul index (0..n-1) to the original feature ID."""
                if 0 <= idx < len(self.all_soul_ids):
                    return self.all_soul_ids[idx]
                return -1
        
        return DistilledSAEMock(data)
    
    @classmethod
    def from_distilled(cls, model_key: str = "llama-3.1-8b-soul", device: str = "cpu") -> "SAEVectorRegistry":
        """
        Create a registry from a predefined distilled SAE configuration.
        
        Args:
            model_key: Key from DISTILLED_SAE_CONFIGS (e.g., "llama-3.1-8b-soul")
            device: Device to load on
            
        Returns:
            SAEVectorRegistry configured with distilled vectors
            
        Example:
            >>> registry = SAEVectorRegistry.from_distilled("llama-3.1-8b-soul")
            >>> v_c = registry.sae.v_c  # Benevolence vector
            >>> v_d = registry.sae.v_d  # Defection vector
        """
        if model_key not in cls.DISTILLED_SAE_CONFIGS:
            available = list(cls.DISTILLED_SAE_CONFIGS.keys())
            raise ValueError(f"Unknown distilled model '{model_key}'. Available: {available}")
        
        config = cls.DISTILLED_SAE_CONFIGS[model_key]
        
        # Check if file exists, if not provide helpful message
        if not Path(config.path).exists():
            logger.warning(
                f"Distilled SAE file not found: {config.path}\n"
                f"Run 'python scripts/distill_llama_soul.py' to generate it."
            )
        
        return cls(
            release=f"distilled-{model_key}",
            sae_type="distilled",
            device=device,
            distilled_path=config.path,
        )
    
    def load(self) -> "SAEVectorRegistry":
        """
        Load the SAE using sae_lens.
        
        Returns:
            Self for method chaining
        """
        try:
            from sae_lens import SAE, HookedSAETransformer
            
            logger.info(f"Loading SAE: {self.release}")
            
            # Load SAE
            self.sae = SAE.from_pretrained(
                release=self.release,
                device=self.device,
            )
            
            self.n_features = self.sae.cfg.neuron_dim
            
            logger.info(f"SAE loaded: {self.n_features} features")
            logger.info(f"  - SAE type: {self.sae.cfg.model_name}")
            logger.info(f"  - Architecture: {self.sae.cfg.architecture}")
            
            return self
            
        except ImportError:
            logger.warning("sae_lens not installed. Using mock SAE.")
            self._setup_mock_sae()
            return self
        except Exception as e:
            logger.warning(f"Failed to load SAE: {e}. Using mock SAE.")
            self._setup_mock_sae()
            return self
    
    def _setup_mock_sae(self):
        """Set up a mock SAE for testing without sae_lens."""
        # Mock SAE with random vectors for testing
        self.n_features = 65536  # Typical for Gemma Scope
        logger.info(f"Using mock SAE: {self.n_features} features")
    
    def get_feature_vector(
        self,
        feature_id: int,
        layer: Optional[int] = None,
        use_cache: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Get the W_dec vector for a specific SAE feature.

        Args:
            feature_id: The feature ID to retrieve
            layer: Optional layer (for multi-layer SAEs)
            use_cache: Whether to use cached vectors

        Returns:
            Feature vector (d_model,), or None if not available
        """
        # No SAE loaded → cannot serve real vectors
        if self.sae is None:
            return None

        if self.n_features == 0 or feature_id < 0 or feature_id >= self.n_features:
            logger.debug(f"Feature ID {feature_id} out of range [0, {self.n_features})")
            return None
        
        # Check cache
        cache_key = feature_id
        if use_cache and cache_key in self._vector_cache:
            return self._vector_cache[cache_key]
        
        # Get vector from SAE
        vector = self._retrieve_feature_vector(feature_id)
        
        if vector is not None and use_cache:
            self._vector_cache[cache_key] = vector
        
        return vector
    
    def get_soul_vectors(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the distilled soul vectors (v_c, v_d) if available.
        
        Returns:
            Tuple of (v_c, v_d) tensors, or None if not a distilled registry
        """
        if self._distilled_data is None:
            return None
        
        v_c = self._distilled_data.get("v_c")
        v_d = self._distilled_data.get("v_d")
        
        if v_c is None or v_d is None:
            return None
            
        return v_c, v_d
    
    def _retrieve_feature_vector(
        self,
        feature_id: int,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve a feature vector from the SAE.
        
        The vector can be either:
        - The decoder weights (what the feature represents)
        - The encoder weights (what activates the feature)
        - The feature activations (for specific inputs)
        
        For steering, we typically use decoder weights.
        """
        if self.sae is None:
            return None

        try:
            # Get decoder weights (W_dec): [n_features, d_model]
            # Each row is the feature's direction in the residual stream
            vector = self.sae.W_dec[feature_id].detach().clone().cpu().float()
            return vector
            
        except Exception as e:
            logger.error(f"Error retrieving feature {feature_id}: {e}")
            return None
    
    def get_features_by_name(
        self,
        name_pattern: str,
    ) -> List[int]:
        """
        Find feature IDs matching a name pattern.
        
        Note: This requires the SAE to have feature explanations,
        which is only available for some releases.
        
        Args:
            name_pattern: Substring to search for
            
        Returns:
            List of matching feature IDs
        """
        # This would require Neuronpedia integration for feature names
        # For now, return empty list
        logger.info(f"Feature name search not implemented without Neuronpedia")
        return []
    
    def batch_get_vectors(
        self,
        feature_ids: List[int],
    ) -> Dict[int, torch.Tensor]:
        """
        Get multiple feature vectors efficiently.
        
        Args:
            feature_ids: List of feature IDs to retrieve
            
        Returns:
            Dictionary mapping feature_id to vector
        """
        vectors = {}
        
        for fid in feature_ids:
            vec = self.get_feature_vector(fid)
            if vec is not None:
                vectors[fid] = vec
        
        return vectors
    
    def get_sae_info(self) -> Dict[str, Any]:
        """Get information about the loaded SAE."""
        if self.sae is None:
            return {
                "release": self.release,
                "type": self.sae_type,
                "n_features": self.n_features,
                "mode": "mock",
            }
        
        info = {
            "release": self.release,
            "type": self.sae_type,
            "n_features": self.n_features,
            "d_in": self.sae.cfg.d_in,
            "d_out": self.sae.cfg.d_out,
            "architecture": self.sae.cfg.architecture,
        }
        
        # Add distilled-specific info
        if self._distilled_data is not None:
            info["mode"] = "distilled"
            info["distilled_path"] = self.distilled_path
            info["model"] = self._distilled_data.get("model", "unknown")
            info["layer"] = self._distilled_data.get("layer", 0)
            info["has_soul_vectors"] = (
                self._distilled_data.get("v_c") is not None and
                self._distilled_data.get("v_d") is not None
            )
        
        return info
    
    def clear_cache(self) -> None:
        """Clear the vector cache."""
        self._vector_cache.clear()
        logger.info("Vector cache cleared")
    
    def enable_cache(self) -> None:
        """Enable vector caching."""
        self._cache_enabled = True
    
    def disable_cache(self) -> None:
        """Disable vector caching."""
        self._cache_enabled = False
        self.clear_cache()
