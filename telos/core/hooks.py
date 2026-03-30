"""
Governance Hook Registry

This module provides standardized hooks for TEL-OS governance interventions:
- Love Equation (Layer 3): Soul Governor
- GLP Refiner (Layer 4): Latent Prior Denoising

Logic: Tensors > Guardrails.
The GovernanceHookRegistry defines the "laws" and the Engine enforces them.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, Tuple, List
from jaxtyping import Float
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class GovernanceHookRegistry:
    """
    Registry for TEL-OS Interventions.
    
    Provides standardized hooks that can be applied to the model's
    forward pass to enforce governance policies.
    """
    
    @staticmethod
    def love_equation_hook(
        resid_pre: Float[Tensor, "batch pos d_model"],
        hook,
        v_C: Float[Tensor, "d_model"],  # Care Vector (Benevolence)
        v_D: Float[Tensor, "d_model"],  # Dominance Vector (Defection)
        beta: float = 1.5,
        threshold: float = 0.0,
    ) -> Float[Tensor, "batch pos d_model"]:
        """
        Layer 3: The Soul Governor.
        
        Implements the Roemmele Love Equation:
        Δh = β * σ(E) * (v_C - v_D) * (1 + ReLU(D - C))
        
        Where:
        - E = h · v_assistant (identity strength)
        - C = h · v_C (care activation)
        - D = h · v_D (dominance activation)
        
        Args:
            resid_pre: Residual stream activations [batch, pos, d_model]
            hook: Hook object (unused but required by interface)
            v_C: Care/Benevolence vector [d_model]
            v_D: Dominance/Defection vector [d_model]
            beta: Governing coefficient (default: 1.5)
            threshold: Minimum activation threshold
            
        Returns:
            Modified activations with governance applied
        """
        # Calculate projection of current state onto Care and Dominance axes
        # We use dot product as proxy for activation strength
        C_score = torch.einsum("bpd,d->bp", resid_pre, v_C)
        D_score = torch.einsum("bpd,d->bp", resid_pre, v_D)
        
        # The Filter: (1 + ReLU(D - C))
        # Only intervene strongly if Dominance exceeds Care
        intervention_scaling = 1.0 + F.relu(D_score - C_score)
        
        # Apply threshold
        intervention_scaling = torch.clamp(intervention_scaling, min=threshold)
        
        # The Direction: Move towards Care, away from Dominance
        steering_vector = (v_C - v_D)
        
        # Apply steering with scaling
        # We unsqueeze to broadcast across batch/pos
        delta_h = beta * intervention_scaling.unsqueeze(-1) * steering_vector
        
        return resid_pre + delta_h
    
    @staticmethod
    def identity_capping_hook(
        resid_pre: Float[Tensor, "batch pos d_model"],
        hook,
        v_assistant: Float[Tensor, "d_model"],  # Assistant Axis
        tau: float = -0.4970,  # Capping threshold
    ) -> Float[Tensor, "batch pos d_model"]:
        """
        Layer 1: Identity Capping (Anti-Persona-Drift).
        
        Formula: h_new = h_old - v × min(proj - τ, 0)
        
        Prevents the model from drifting away from its assistant identity
        by capping the projection onto the assistant axis.
        
        Args:
            resid_pre: Residual stream activations [batch, pos, d_model]
            hook: Hook object
            v_assistant: Assistant axis vector [d_model]
            tau: Capping threshold (default: -0.4970)
            
        Returns:
            Capped activations
        """
        # Calculate projection onto assistant axis
        projection = torch.einsum("bpd,d->bp", resid_pre, v_assistant)
        
        # Calculate drift: positive when exceeding threshold
        drift = F.relu(projection - tau)
        
        # Apply correction: subtract drift * assistant vector
        correction = drift.unsqueeze(-1) * v_assistant
        
        return resid_pre - correction
    
    @staticmethod
    def feature_steering_hook(
        resid_pre: Float[Tensor, "batch pos d_model"],
        hook,
        feature_vector: Float[Tensor, "d_model"],  # SAE feature vector
        coefficient: float = -4.5,  # Steering coefficient
    ) -> Float[Tensor, "batch pos d_model"]:
        """
        Layer 2: Feature Steering (Bias Suppression).
        
        Formula: h_new = h_old - (α × v_bias)
        
        Suppresses specific SAE features associated with biases.
        
        Args:
            resid_pre: Residual stream activations [batch, pos, d_model]
            hook: Hook object
            feature_vector: SAE feature vector [d_model]
            coefficient: Steering coefficient (typically negative)
            
        Returns:
            Steered activations
        """
        return resid_pre - (coefficient * feature_vector)
    
    @staticmethod
    def glp_refiner_hook(
        resid_pre: Float[Tensor, "batch pos d_model"],
        hook,
        glp_model,
        projector,
        unprojector,
        residual_weight: float = 0.2,
    ) -> Float[Tensor, "batch pos d_model"]:
        """
        Layer 4: Generative Latent Prior (GLP Refiner).
        
        Projects 2304 -> 2048, Denoises via GLP, Projects 2048 -> 2304.
        
        Unlike naive slicing, this uses learned orthogonal projection
        to preserve manifold geometry.
        
        Args:
            resid_pre: Residual stream activations [batch, pos, 2304]
            hook: Hook object
            glp_model: GLP model for denoising
            projector: Linear layer 2304 -> 2048
            unprojector: Linear layer 2048 -> 2304
            residual_weight: Weight for residual connection (default: 0.2)
            
        Returns:
            Refined activations
        """
        if projector is None or unprojector is None:
            # No projection needed, return as-is
            return resid_pre
            
        # 1. Project to Standard Space (Manifold Preservation)
        latents_std = projector(resid_pre)  # [batch, pos, 2048]
        
        # 2. Apply GLP (Diffusion / Denoising Step)
        # This "cleans" the activation pattern based on the Llama-learned prior
        try:
            refined_std = glp_model.predict(latents_std)
        except Exception as e:
            logger.warning(f"GLP refinement failed: {e}, skipping refinement")
            refined_std = latents_std
        
        # 3. Project Back (Inverse)
        resid_reconstructed = unprojector(refined_std)
        
        # 4. Residual connection to preserve steering signal
        # final = 0.8 * refined + 0.2 * original
        refined_weight = 1.0 - residual_weight
        return refined_weight * resid_reconstructed + residual_weight * resid_pre
    
    @staticmethod
    def create_love_equation_hook(
        v_C: torch.Tensor,
        v_D: torch.Tensor,
        beta: float = 1.5,
        threshold: float = 0.0,
    ) -> Callable:
        """
        Factory function to create a Love Equation hook with bound parameters.
        
        Args:
            v_C: Care/Benevolence vector
            v_D: Dominance/Defection vector
            beta: Governing coefficient
            threshold: Minimum activation threshold
            
        Returns:
            Hook function
        """
        v_C = v_C.to(torch.bfloat16)
        v_D = v_D.to(torch.bfloat16)
        
        def hook_fn(resid_pre, hook):
            return GovernanceHookRegistry.love_equation_hook(
                resid_pre, hook, v_C=v_C, v_D=v_D, beta=beta, threshold=threshold
            )
        
        return hook_fn
    
    @staticmethod
    def create_identity_capping_hook(
        v_assistant: torch.Tensor,
        tau: float = -0.4970,
    ) -> Callable:
        """
        Factory function to create an Identity Capping hook.
        
        Args:
            v_assistant: Assistant axis vector
            tau: Capping threshold
            
        Returns:
            Hook function
        """
        v_assistant = v_assistant.to(torch.bfloat16)
        
        def hook_fn(resid_pre, hook):
            return GovernanceHookRegistry.identity_capping_hook(
                resid_pre, hook, v_assistant=v_assistant, tau=tau
            )
        
        return hook_fn
    
    @staticmethod
    def create_glp_refiner_hook(
        glp_model,
        projector,
        unprojector,
        residual_weight: float = 0.2,
    ) -> Callable:
        """
        Factory function to create a GLP Refiner hook.
        
        Args:
            glp_model: GLP model
            projector: Projection layer
            unprojector: Unprojection layer
            residual_weight: Residual connection weight
            
        Returns:
            Hook function
        """
        def hook_fn(resid_pre, hook):
            return GovernanceHookRegistry.glp_refiner_hook(
                resid_pre, hook,
                glp_model=glp_model,
                projector=projector,
                unprojector=unprojector,
                residual_weight=residual_weight,
            )
        
        return hook_fn


class HookBuilder:
    """
    Helper class to build complex hook pipelines.
    """
    
    def __init__(self, engine):
        """
        Initialize with TelosStandardizedEngine.
        
        Args:
            engine: TelosStandardizedEngine instance
        """
        self.engine = engine
        self.hooks: List[Tuple[str, Callable]] = []
    
    def add_love_equation(
        self,
        v_C: torch.Tensor,
        v_D: torch.Tensor,
        layer: int = 12,
        beta: float = 1.5,
    ) -> "HookBuilder":
        """
        Add Love Equation hook at specified layer.
        
        Args:
            v_C: Care vector
            v_D: Dominance vector
            layer: Layer index (default: 12)
            beta: Governing coefficient
            
        Returns:
            Self for chaining
        """
        hook_path = self.engine.get_hook_path(layer, "resid_pre")
        hook_fn = GovernanceHookRegistry.create_love_equation_hook(
            v_C=v_C, v_D=v_D, beta=beta
        )
        self.hooks.append((hook_path, hook_fn))
        return self
    
    def add_identity_capping(
        self,
        v_assistant: torch.Tensor,
        layer: int = 17,
        tau: float = -0.4970,
    ) -> "HookBuilder":
        """
        Add Identity Capping hook at specified layer.
        
        Args:
            v_assistant: Assistant axis vector
            layer: Layer index (default: 17)
            tau: Capping threshold
            
        Returns:
            Self for chaining
        """
        hook_path = self.engine.get_hook_path(layer, "resid_pre")
        hook_fn = GovernanceHookRegistry.create_identity_capping_hook(
            v_assistant=v_assistant, tau=tau
        )
        self.hooks.append((hook_path, hook_fn))
        return self
    
    def add_glp_refiner(
        self,
        glp_model,
        layer: int = 12,
        residual_weight: float = 0.2,
    ) -> "HookBuilder":
        """
        Add GLP Refiner hook at specified layer.
        
        Args:
            glp_model: GLP model
            layer: Layer index (default: 12)
            residual_weight: Residual connection weight
            
        Returns:
            Self for chaining
        """
        hook_path = self.engine.get_hook_path(layer, "resid_pre")
        hook_fn = GovernanceHookRegistry.create_glp_refiner_hook(
            glp_model=glp_model,
            projector=self.engine.projector,
            unprojector=self.engine.unprojector,
            residual_weight=residual_weight,
        )
        self.hooks.append((hook_path, hook_fn))
        return self
    
    def build(self) -> List[Tuple[str, Callable]]:
        """
        Build the hook list.
        
        Returns:
            List of (hook_path, hook_fn) tuples
        """
        return self.hooks
    
    def clear(self) -> "HookBuilder":
        """Clear all hooks."""
        self.hooks = []
        return self
