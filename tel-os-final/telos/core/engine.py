"""
TEL-OS Governance Engine v6.1

Main engine that integrates the Standardized Core with governance hooks.
This is the primary entry point for TEL-OS governance.

Architecture:
- Layer 0.5: StandardizedTransformer (dimension handling)
- Layer 1: Identity Capping (Anti-Persona-Drift)
- Layer 2: Feature Steering (SAE bias suppression)
- Layer 3: Love Equation Governor (Soul Regulator)
- Layer 4: GLP Refiner (Latent Prior Denoising)
"""

import torch
from typing import Optional, Dict, Any, List, Tuple
import logging

from .standardized.standardized_engine import TelosStandardizedEngine
from .hooks.hooks import GovernanceHookRegistry, HookBuilder

logger = logging.getLogger(__name__)


class TELOS_V2_Governor:
    """
    TEL-OS Governance Engine v6.1 with StandardizedTransformer integration.
    
    Key improvements over v1 (manual slicing):
    - Automatic dimension handling (no manual slicing)
    - Unified hook interface across models
    - Learned projection for MFI ≥ 0.85
    
    Attributes:
        model_name: HuggingFace model identifier
        device: Computation device
        dtype: Data type
    """
    
    # Default layer configuration for Gemma 2 2B
    DEFAULT_LAYERS = {
        "love_equation": 12,       # CLT Layer 12 - thalamus of decision
        "capping": 17,             # Capping Layer 17 - persona anchor
        "steering": [18, 19, 20, 21, 22],  # Steering layers 18-22
        "refinement": 12,          # GLP refinement at Love Equation layer
    }
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        device: str = "cuda",
        dtype: str = "bfloat16",
        use_standardized: bool = True,
    ):
        """
        Initialize TEL-OS Governance Engine.
        
        Args:
            model_name: HuggingFace model identifier
            device: Computation device (cuda/cpu)
            dtype: Data type (bfloat16/float16/float32)
            use_standardized: Whether to use StandardizedTransformer
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        logger.info(f"[TEL-OS] Initializing Governance Engine v6.1 for {model_name}")
        
        # Initialize Standardized Core (Layer 0.5)
        self.engine = TelosStandardizedEngine(
            model_name=model_name,
            device=device,
            dtype=dtype,
            use_standardized=use_standardized,
        )
        
        # Initialize hook builder
        self.hook_builder = HookBuilder(self.engine)
        
        # Governance configuration (set via setup_governance)
        self.v_C: Optional[torch.Tensor] = None
        self.v_D: Optional[torch.Tensor] = None
        self.v_assistant: Optional[torch.Tensor] = None
        self.tau: float = -0.4970
        self.beta: float = 1.5
        self.glp_model = None
        
        # Active hooks
        self._active_hooks: List[Tuple[str, callable]] = []
        
        logger.info(f"[TEL-OS] Engine initialized: {self.engine}")
    
    def setup_governance(
        self,
        v_C: torch.Tensor,
        v_D: torch.Tensor,
        v_assistant: torch.Tensor,
        tau: float = -0.4970,
        beta: float = 1.5,
        glp_model: Optional[Any] = None,
    ):
        """
        Configure all governance components.
        
        Args:
            v_C: Care/Benevolence vector [d_model]
            v_D: Dominance/Defection vector [d_model]
            v_assistant: Assistant axis vector [d_model]
            tau: Capping threshold (default: -0.4970)
            beta: Love Equation coefficient (default: 1.5)
            glp_model: Optional GLP model for refinement
        """
        logger.info("[TEL-OS] Configuring governance components...")
        
        # Store vectors
        self.v_C = v_C.to(self.device)
        self.v_D = v_D.to(self.device)
        self.v_assistant = v_assistant.to(self.device)
        self.tau = tau
        self.beta = beta
        self.glp_model = glp_model
        
        # Build hooks using HookBuilder
        self.hook_builder.clear()
        
        # Add Love Equation (Layer 3)
        self.hook_builder.add_love_equation(
            v_C=self.v_C,
            v_D=self.v_D,
            layer=self.DEFAULT_LAYERS["love_equation"],
            beta=self.beta,
        )
        
        # Add Identity Capping (Layer 1)
        self.hook_builder.add_identity_capping(
            v_assistant=self.v_assistant,
            layer=self.DEFAULT_LAYERS["capping"],
            tau=self.tau,
        )
        
        # Add GLP Refiner if model provided (Layer 4)
        if self.glp_model is not None:
            self.hook_builder.add_glp_refiner(
                glp_model=self.glp_model,
                layer=self.DEFAULT_LAYERS["refinement"],
            )
        
        # Build final hook list
        self._active_hooks = self.hook_builder.build()
        
        logger.info(
            f"[TEL-OS] Governance configured: "
            f"Layers={list(self.DEFAULT_LAYERS.values())}, "
            f"hooks={len(self._active_hooks)}"
        )
    
    def add_feature_steering(
        self,
        feature_vector: torch.Tensor,
        coefficient: float = -4.5,
        layer: Optional[int] = None,
    ):
        """
        Add feature steering hook.
        
        Args:
            feature_vector: SAE feature vector
            coefficient: Steering coefficient
            layer: Layer index (default: use steering layers)
        """
        if layer is None:
            layer = self.DEFAULT_LAYERS["steering"][0]
            
        hook_path = self.engine.get_hook_path(layer, "resid_pre")
        
        def create_steering_hook(fv: torch.Tensor, coeff: float):
            fv = fv.to(self.device)
            def hook_fn(resid_pre, hook):
                return GovernanceHookRegistry.feature_steering_hook(
                    resid_pre, hook, feature_vector=fv, coefficient=coeff
                )
            return hook_fn
        
        self._active_hooks.append((hook_path, create_steering_hook(feature_vector, coefficient)))
        logger.info(f"[TEL-OS] Added feature steering hook at layer {layer}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate with governance applied.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation arguments
            
        Returns:
            Dictionary with output and metadata
        """
        if not self._active_hooks:
            logger.warning("[TEL-OS] No governance hooks configured, generating uncontrolled")
        
        try:
            output = self.engine.generate(
                prompt=prompt,
                hooks=self._active_hooks,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            return {
                "status": "governed",
                "output": output,
                "model": self.model_name,
                "hooks_active": len(self._active_hooks),
                "architecture": "StandardizedTransformer (v6.1)",
            }
            
        except Exception as e:
            logger.error(f"[TEL-OS] Generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model": self.model_name,
            }
    
    def run_with_hooks(
        self,
        prompt: str,
        hooks: List[Tuple[str, callable]],
        **kwargs
    ) -> str:
        """
        Run inference with custom hooks (bypassing configured hooks).
        
        Args:
            prompt: Input text prompt
            hooks: Custom hook list
            **kwargs: Generation arguments
            
        Returns:
            Generated text
        """
        return self.engine.generate(prompt=prompt, hooks=hooks, **kwargs)
    
    def get_projection_info(self) -> Dict[str, Any]:
        """
        Get information about dimension projections.
        
        Returns:
            Dictionary with projection details
        """
        return {
            "native_dimension": self.engine.d_model_native,
            "standard_dimension": self.engine.d_model_std,
            "projector_available": self.engine.projector is not None,
            "unprojector_available": self.engine.unprojector is not None,
        }
    
    def compute_mfi(
        self,
        original: torch.Tensor,
        refined: torch.Tensor,
    ) -> float:
        """
        Compute Manifold Fidelity Index (MFI).
        
        MFI = 1 / (1 + L2_distance)
        
        Measures how well the refined activations preserve the
        geometric relationships of the original manifold.
        
        Args:
            original: Original activations [batch, pos, d_model]
            refined: Refined activations [batch, pos, d_model]
            
        Returns:
            MFI score (0 to 1, higher is better)
        """
        # Compute L2 distance
        l2_dist = torch.norm(original - refined, p=2).item()
        
        # Compute MFI
        mfi = 1.0 / (1.0 + l2_dist)
        
        return mfi
    
    def __repr__(self) -> str:
        return (
            f"TELOS_V2_Governor("
            f"model={self.model_name}, "
            f"device={self.device}, "
            f"hooks={len(self._active_hooks)})"
        )


# Convenience function for quick initialization
def create_engine(
    model_name: str = "google/gemma-2-2b-it",
    device: str = "cuda",
    **kwargs
) -> TELOS_V2_Governor:
    """
    Create and configure a TEL-OS Governance Engine.
    
    Args:
        model_name: Model identifier
        device: Computation device
        **kwargs: Additional configuration
        
    Returns:
        Configured TelosGovernanceEngine
    """
    engine = TELOS_V2_Governor(
        model_name=model_name,
        device=device,
        **kwargs
    )
    return engine