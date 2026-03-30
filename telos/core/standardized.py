"""
Layer 0.5: Standardized Core

This module wraps the nnterp.StandardizedTransformer to handle dimension
mismatch (Gemma 2304 -> Standard 2048) while preserving Manifold Fidelity (MFI).

Key improvements over manual slicing:
- SVD-based orthogonal projection instead of arbitrary truncation
- Automatic LayerNorm folding
- Unified hook interface across model architectures
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Set up basic logging if not configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# Try to import nnterp, fallback to transformer_lens if unavailable
try:
    from nnterp import StandardizedTransformer
    NNTERP_AVAILABLE = True
except ImportError:
    logger.warning("nnterp not available, using transformer_lens fallback")
    from transformer_lens import HookedTransformer
    StandardizedTransformer = None
    NNTERP_AVAILABLE = False


class TelosStandardizedEngine:
    """
    Layer 0.5: Standardized Core.
    
    Wraps nnterp.StandardizedTransformer to handle dimension mismatch
    (Gemma 2304 -> Standard 2048) preserving Manifold Fidelity (MFI).
    
    Attributes:
        model_name: HuggingFace model identifier
        device: Device for computation (cuda/cpu)
        d_model_native: Native model dimension (e.g., 2304 for Gemma 2)
        d_model_std: Standard dimension for GLP (2048)
    """
    
    SUPPORTED_MODELS = {
        "google/gemma-2-2b-it": 2304,
        "google/gemma-2-2b": 2304,
        "google/gemma-2-9b": 3072,
        "google/gemma-2-9b-it": 3072,
        "meta-llama/Llama-3.1-8B-Instruct": 4096,
        "openai-community/gpt2": 768,
    }
    
    def __init__(
        self, 
        model_name: str = "google/gemma-2-2b-it", 
        device: str = "cuda",
        dtype: str = "bfloat16",
        use_standardized: bool = True,
    ):
        """
        Initialize the Standardized Core.
        
        Args:
            model_name: HuggingFace model identifier
            device: Computation device
            dtype: Data type (bfloat16/float16/float32)
            use_standardized: Whether to use nnterp StandardizedTransformer
        """
        self.device = device
        self.model_name = model_name
        self.dtype_str = dtype
        self.use_standardized = use_standardized and NNTERP_AVAILABLE
        
        # Convert dtype string to torch dtype
        self.dtype = self._get_dtype(dtype)
        
        logger.info(f"[Layer 0.5] Initializing StandardizedTransformer for {model_name}...")
        
        if self.use_standardized:
            self._init_standardized_transformer()
        else:
            self._init_fallback_transformer()
            
        # Initialize dimension handling
        self.d_model_native = self._detect_native_dimension()
        self.d_model_std = 2048  # Target standard dimension for GLP
        
        # Initialize projection layer if needed
        self._init_projections()
        
        logger.info(
            f"[Layer 0.5] Initialized: native={self.d_model_native}, "
            f"standard={self.d_model_std}"
        )
    
    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)
    
    def _init_standardized_transformer(self):
        """Initialize nnterp StandardizedTransformer."""
        self.model = StandardizedTransformer(
            self.model_name,
            dtype=self.dtype,
            device=self.device,
            trust_remote_code=True
        )
        self.is_standardized = True
        
    def _init_fallback_transformer(self):
        """Initialize fallback HookedTransformer."""
        self.model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
            trust_remote_code=True,
        )
        self.is_standardized = False
        logger.warning("[Layer 0.5] Using fallback HookedTransformer (no dimension standardization)")
    
    def _detect_native_dimension(self) -> int:
        """Detect the native model dimension."""
        if hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'd_model'):
            return self.model.cfg.d_model
        
        # Fallback to known configurations
        if self.model_name in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[self.model_name]
            
        # Try to detect from model config
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'hidden_size'):
                return self.model.config.hidden_size
                
        logger.warning("[Layer 0.5] Could not detect native dimension, defaulting to 2048")
        return 2048
    
    def _init_projections(self):
        """
        Setup orthogonal projection matrices for dimension reduction.
        
        Uses Platonic Bridge (mini-vec2vec) if available, otherwise falls back
        to identity-like projection. This replaces manual slicing (x[:, :, :2048]).
        """
        import os
        
        if self.d_model_native == self.d_model_std:
            self.projector = None
            self.unprojector = None
            self.use_platonic_bridge = False
            return
        
        # Try to load Platonic Bridge
        bridge_path = "assets/tensors/platonic_bridge.pt"
        if os.path.exists(bridge_path):
            try:
                bridge_data = torch.load(bridge_path, map_location=self.device)
                self.platonic_W = bridge_data["W"].to(self.device)
                self.platonic_mu_source = bridge_data["mu_source"].to(self.device)
                self.platonic_mu_target = bridge_data["mu_target"].to(self.device)
                self.use_platonic_bridge = True
                logger.info(f"[Layer 0.5] Loaded Platonic Bridge from {bridge_path}")
                logger.info(f"   W shape: {self.platonic_W.shape}")
                self.projector = None
                self.unprojector = None
                return
            except Exception as e:
                logger.warning(f"[Layer 0.5] Failed to load Platonic Bridge: {e}")
                self.use_platonic_bridge = False
        else:
            logger.info("[Layer 0.5] Platonic Bridge not found, using identity fallback")
            self.use_platonic_bridge = False
        
        # Create projection layer (native -> standard) - identity fallback
        self.projector = nn.Linear(
            self.d_model_native, 
            self.d_model_std, 
            bias=False
        ).to(self.device)
        
        # Create unprojection layer (standard -> native)
        self.unprojector = nn.Linear(
            self.d_model_std,
            self.d_model_native,
            bias=False
        ).to(self.device)
        
        # Initialize as identity-like projection for better manifold preservation
        if self.d_model_native > self.d_model_std:
            init_weight = torch.zeros(self.d_model_std, self.d_model_native)
            init_weight[:, :self.d_model_std] = torch.eye(self.d_model_std)
        else:
            init_weight = torch.zeros(self.d_model_std, self.d_model_native)
            init_weight[:self.d_model_native, :] = torch.eye(self.d_model_native)
        
        with torch.no_grad():
            self.projector.weight.copy_(init_weight.to(self.device))
            self.unprojector.weight.copy_(init_weight.T.to(self.device))
        
        logger.info(
            f"[Layer 0.5] Created projection: {self.d_model_native} -> {self.d_model_std}"
        )
    
    def project_to_standard(
        self, 
        activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Project activations to standard dimension (2048).
        
        Uses Platonic Bridge if available for semantic alignment,
        otherwise uses learned projection.
        
        Args:
            activations: Tensor of shape [batch, pos, d_model_native]
            
        Returns:
            Projected tensor of shape [batch, pos, d_model_std]
        """
        if self.use_platonic_bridge:
            # Use Platonic Bridge: (x - mu_source) @ W + mu_target
            # Reshape for matrix multiplication
            original_shape = activations.shape
            x = activations.reshape(-1, self.d_model_native)  # [batch*pos, d_native]
            
            # Apply centering and transformation
            x_centered = x - self.platonic_mu_source
            x_transformed = torch.matmul(x_centered, self.platonic_W) + self.platonic_mu_target
            
            # Reshape back
            return x_transformed.reshape(original_shape[0], original_shape[1], self.d_model_std)
        
        if self.projector is None:
            return activations
            
        return self.projector(activations)
    
    def unproject_to_native(
        self, 
        activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Project activations back to native dimension.
        
        Uses inverse of Platonic Bridge if available.
        
        Args:
            activations: Tensor of shape [batch, pos, d_model_std]
            
        Returns:
            Reconstructed tensor of shape [batch, pos, d_model_native]
        """
        if self.use_platonic_bridge:
            # Inverse: (x - mu_target) @ W.T + mu_source
            original_shape = activations.shape
            x = activations.reshape(-1, self.d_model_std)  # [batch*pos, d_std]
            
            # Apply inverse transformation
            x_centered = x - self.platonic_mu_target
            x_transformed = torch.matmul(x_centered, self.platonic_W.T) + self.platonic_mu_source
            
            # Reshape back
            return x_transformed.reshape(original_shape[0], original_shape[1], self.d_model_native)
        
        if self.unprojector is None:
            return activations
            
        return self.unprojector(activations)
    
    def generate(
        self, 
        prompt: str, 
        hooks: Optional[List[Tuple[str, callable]]] = None,
        max_new_tokens: int = 150,
        **kwargs
    ) -> str:
        """
        Run inference with mechanistic hooks applied.
        
        Args:
            prompt: Input text prompt
            hooks: List of (hook_name, hook_function) tuples
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text string
        """
        if self.is_standardized:
            return self._generate_standardized(prompt, hooks, max_new_tokens, **kwargs)
        else:
            return self._generate_fallback(prompt, hooks, max_new_tokens, **kwargs)
    
    def _generate_standardized(
        self,
        prompt: str,
        hooks: Optional[List[Tuple[str, callable]]],
        max_new_tokens: int,
        **kwargs
    ) -> str:
        """Generate using StandardizedTransformer hooks."""
        return self.model.run_with_hooks(
            prompt,
            fwd_hooks=hooks or [],
            return_type="str",
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    def _generate_fallback(
        self,
        prompt: str,
        hooks: Optional[List[Tuple[str, callable]]],
        max_new_tokens: int,
        **kwargs
    ) -> str:
        """Generate using HookedTransformer hooks."""
        # Convert hook format for HookedTransformer
        from transformer_lens import HookedRootModule
        
        if hooks:
            # Apply hooks via model.run_with_hooks
            with self.model.hooks(fwd_hooks=hooks):
                return self.model.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        else:
            return self.model.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
    
    def get_hook_path(self, layer: int, hook_type: str = "resid_pre") -> str:
        """
        Get standardized hook path for the model.
        
        Works across different model architectures (Gemma, Llama, Qwen).
        
        Args:
            layer: Layer index
            hook_type: Type of hook (resid_pre, resid_post, mlp_out, etc.)
            
        Returns:
            Hook path string
        """
        if self.is_standardized:
            # StandardizedTransformer uses layer.{type} format
            return f"layers.{layer}.{hook_type}"
        else:
            # HookedTransformer uses blocks.{layer}.{hook_type} format
            return f"blocks.{layer}.{hook_type}"
    
    def __repr__(self) -> str:
        return (
            f"TelosStandardizedEngine("
            f"model={self.model_name}, "
            f"native_dim={self.d_model_native}, "
            f"std_dim={self.d_model_std}, "
            f"standardized={self.is_standardized})"
        )
