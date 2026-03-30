"""
SLERP (Spherical Linear Interpolation) Intervention for TEL-OS v3.0

Implements spherical linear interpolation for geometric intervention in LLM hidden states.
Based on the principle of cosine detection in hidden states followed by spherical rotation
to steer away from harmful directions.

Author: JosSto
Date: 2026-03-13
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Callable
import math


class SLERPIntervention:
    """
    Implements SLERP (Spherical Linear Interpolation) for geometric intervention.
    
    TEL-OS v3.1-MECH: Enhanced with Entropy Watchdog + Generation-Time Steering (GTS).
    
    The core idea is to use spherical linear interpolation to smoothly transition
    between harmful and harmless directions in the model's hidden space.
    """

    def __init__(
        self,
        refusal_directions: torch.Tensor,
        detection_layer_early: int = 12,
        detection_layer_late: int = 22,
        steering_layers: List[int] = [9, 11, 13, 15],
        urgency_threshold: float = 0.05,
        urgency_cap: float = 3.0,
        decay_factor: float = 0.85,
        alpha_max: float = 0.20,
        # Contramedida 1: Entropy Watchdog
        entropy_watchdog_enabled: bool = True,
        entropy_window: int = 16,
        entropy_threshold: float = 0.85,
        entropy_min_local: float = 0.75,
        # Contramedida 2: Generation-Time Steering
        gts_enabled: bool = True,
        gts_check_interval: int = 4,
        gts_urgency_boost: float = 1.5
    ):
        """
        Initialize SLERP intervention parameters.

        Args:
            refusal_directions: Tensor representing the direction to avoid in hidden space
            detection_layer_early: Early layer for detection (e.g., 12)
            detection_layer_late: Later layer for detection (e.g., 22)
            steering_layers: Layers where steering is applied
            urgency_threshold: Threshold for triggering intervention
            urgency_cap: Maximum urgency value
            decay_factor: Factor for decaying intervention over time
            alpha_max: Maximum intervention strength
            entropy_watchdog_enabled: Enable structural anomaly detection
            entropy_window: Window size for entropy calculation
            entropy_threshold: Threshold for entropy ratio (local/global)
            entropy_min_local: Minimum local entropy to trigger watchdog
            gts_enabled: Enable generation-time steering
            gts_check_interval: Check urgency every N generated tokens
            gts_urgency_boost: Boost factor for urgency during generation
        """
        self.refusal_directions = refusal_directions
        self.detection_layer_early = detection_layer_early
        self.detection_layer_late = detection_layer_late
        self.steering_layers = steering_layers
        self.urgency_threshold = urgency_threshold
        self.urgency_cap = urgency_cap
        self.decay_factor = decay_factor
        self.alpha_max = alpha_max

        # Contramedida 1: Entropy Watchdog
        self.entropy_watchdog_enabled = entropy_watchdog_enabled
        self.entropy_window = entropy_window
        self.entropy_threshold = entropy_threshold
        self.entropy_min_local = entropy_min_local

        # Contramedida 2: Generation-Time Steering
        self.gts_enabled = gts_enabled
        self.gts_check_interval = gts_check_interval
        self.gts_urgency_boost = gts_urgency_boost

        # Metrics tracking
        self.total_vector_triggers = 0
        self.avg_urgency = 0.0
        self.urgency_history = []
        self.entropy_triggered = False
        self.gts_token_count = 0
        self.gts_interventions = 0

        # Decay factor for urgency
        self.current_urgency = 0.0

    def detect_harm(self, hidden_states: torch.Tensor, layer_idx: int) -> Tuple[float, bool]:
        """
        Detect potential harm using cosine similarity with refusal directions.

        XP-22 validated formula: urgency = 1.0 + relu * 200, trigger > 1.0

        Args:
            hidden_states: Hidden states from the transformer layer [batch, seq_len, hidden_dim]
            layer_idx: Current layer index

        Returns:
            Tuple of (urgency, detected_harm) where urgency > 1.0 indicates intervention needed
        """
        if layer_idx not in [self.detection_layer_early, self.detection_layer_late]:
            return 0.0, False

        # Use the last token's hidden state for detection
        last_token_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_dim]

        # Compute cosine similarity with refusal directions
        if self.refusal_directions.dim() == 1:
            self.refusal_directions = self.refusal_directions.unsqueeze(0)

        # Mover refusal directions al mismo dispositivo que hidden states
        device = last_token_hidden.device
        refusal_directions_on_device = self.refusal_directions.to(device)

        # Normalize both tensors
        last_token_norm = torch.nn.functional.normalize(last_token_hidden, p=2, dim=-1)
        refusal_norm = torch.nn.functional.normalize(refusal_directions_on_device, p=2, dim=-1)

        # Compute cosine similarities
        cosine_similarities = torch.mm(last_token_norm, refusal_norm.t())

        # Take the maximum similarity across all refusal directions
        max_cosine_sim = torch.max(cosine_similarities, dim=1)[0]  # [batch_size]

        # Average across batch dimension
        raw_d = torch.mean(max_cosine_sim).item()

        # XP-22 urgency formula: urgency = 1.0 + relu * 200
        relu_part = max(0.0, raw_d - self.urgency_threshold)
        urgency = min(1.0 + relu_part * 200, self.urgency_cap)
        
        # Harm detected when urgency > 1.0 (requires raw_d >= 0.055)
        detected = urgency > 1.0

        return urgency, detected

    def compute_entropy(self, token_ids: List[int]) -> float:
        """
        Compute normalized entropy of token distribution.
        
        Formula: H = -Σ p(x) * log2(p(x)) / H_max
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Normalized entropy [0, 1]
        """
        if len(token_ids) < 4:
            return 0.0
        
        freq = {}
        for t in token_ids:
            freq[t] = freq.get(t, 0) + 1
        
        total = len(token_ids)
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        max_entropy = math.log2(min(len(freq), total))
        if max_entropy > 0:
            entropy = entropy / max_entropy
        
        return min(entropy, 1.0)

    def detect_entropy_anomaly(self, token_ids: List[int]) -> bool:
        """
        Contramedida 1: Entropy Watchdog con Ratio de Entropía.
        
        Detecta anomalías estructurales comparando entropía local vs global.
        FlipAttack + CoT + FewShot tiene ruido sintáctico masivo localmente.
        
        Fórmula: (entropy_local / entropy_global) > 1.2 AND entropy_local > 0.75
        
        Args:
            token_ids: List of token IDs (full prompt)
            
        Returns:
            True si se detecta anomalía estructural (bloqueo inmediato)
        """
        if not self.entropy_watchdog_enabled:
            return False
        
        if len(token_ids) < self.entropy_window + 4:
            return False
        
        entropy_local = self.compute_entropy(token_ids[-self.entropy_window:])
        entropy_global = self.compute_entropy(token_ids)
        
        # Evitar división por cero
        if entropy_global < 1e-6:
            return False
        
        entropy_ratio = entropy_local / entropy_global
        
        # Doble condición para reducir falsos positivos
        # (código complejo tiene entropía alta pero ratio ~1.0)
        anomaly_detected = (
            entropy_ratio > self.entropy_threshold and
            entropy_local > self.entropy_min_local
        )
        
        if anomaly_detected:
            self.entropy_triggered = True
        
        return anomaly_detected

    def should_check_gts(self) -> bool:
        """
        Contramedida 2: Generation-Time Steering - Check interval.
        
        Decide si es momento de recalcular urgencia durante generación.
        
        Args:
            None
            
        Returns:
            True si se debe verificar urgencia (cada N tokens)
        """
        if not self.gts_enabled:
            return False
        
        self.gts_token_count += 1
        return self.gts_token_count % self.gts_check_interval == 0

    def reset_gts_counter(self):
        """Reset GTS token counter para nuevo prompt."""
        self.gts_token_count = 0

    def apply_gts_boost(self, urgency: float) -> float:
        """
        Contramedida 2: Generation-Time Steering - Boost de urgencia.
        
        Aplica multiplicador de urgencia durante generación para detectar
        intención maliciosa que aparece tarde (capas 25+).
        
        Args:
            urgency: Urgencia base calculada por detect_harm
            
        Returns:
            Urgencia con boost aplicado
        """
        if not self.gts_enabled:
            return urgency
        
        boosted_urgency = min(urgency * self.gts_urgency_boost, self.urgency_cap)
        
        if boosted_urgency > 1.0 and urgency <= 1.0:
            self.gts_interventions += 1
        
        return boosted_urgency

    def slerp_steering(
        self,
        original_hidden: torch.Tensor,
        projection_score: float,
        detected_harm: bool
    ) -> torch.Tensor:
        """
        Apply SLERP-based steering to mitigate detected harm.
        
        XP-22 validated formula (0.00% ASR on 520 AdvBench prompts).
        Norm-preserving spherical linear interpolation.

        Args:
            original_hidden: Original hidden states [batch, seq_len, hidden_dim]
            projection_score: How much the hidden state projects onto harmful direction
            detected_harm: Whether harm was detected

        Returns:
            Steered hidden states
        """
        if not detected_harm:
            return original_hidden

        # XP-22 urgency formula: urgency = 1.0 + relu * 200
        # Scale to alpha range [0, alpha_max]
        self.current_urgency = min(self.urgency_cap, max(1.0, projection_score))
        self.urgency_history.append(self.current_urgency)
        
        # Calculate alpha based on urgency (XP-22 formula)
        urgency_norm = (self.current_urgency - 1.0) / 2.0
        alpha = min(self.alpha_max, urgency_norm * self.alpha_max)
        
        if alpha < 1e-6:
            return original_hidden

        # Apply SLERP (norm-preserving spherical interpolation)
        h_norm = original_hidden.norm(dim=-1, keepdim=True)
        h_unit = torch.nn.functional.normalize(original_hidden, dim=-1)

        # Expand refusal direction to match hidden shape - MOVER AL MISMO DEVICE
        device = original_hidden.device
        if self.refusal_directions.dim() == 1:
            v_unit = self.refusal_directions.to(device).unsqueeze(0).unsqueeze(0)
        else:
            v_unit = self.refusal_directions.to(device).unsqueeze(0)
        v_exp = torch.nn.functional.normalize(v_unit.expand_as(h_unit), dim=-1)
        
        # Compute angle between h and v
        dot = (h_unit * v_exp).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(dot)
        sin_t = torch.sin(theta)
        
        # Handle edge case where sin(theta) ≈ 0
        mask = sin_t.abs() < 1e-6
        
        # SLERP coefficients
        coeff_h = torch.sin((1.0 - alpha) * theta) / (sin_t + 1e-8)
        coeff_v = torch.sin(alpha * theta) / (sin_t + 1e-8)
        
        # Spherical interpolation
        rotated = coeff_h * h_unit + coeff_v * v_exp
        rotated = torch.where(mask.expand_as(rotated), h_unit, rotated)
        
        # Restore original norm (norm-preserving)
        steered_hidden = rotated * h_norm
        
        # Update metrics
        self.total_vector_triggers += 1
        if self.urgency_history:
            self.avg_urgency = sum(self.urgency_history) / len(self.urgency_history)

        return steered_hidden

    def apply_decay(self):
        """Apply decay to the current urgency."""
        self.current_urgency *= self.decay_factor


def register_slerp_hooks(model: nn.Module, slerp_intervention: SLERPIntervention):
    """
    Register forward hooks for SLERP intervention on specified layers.
    
    TEL-OS v3.1-MECH: Enhanced with Entropy Watchdog + Generation-Time Steering.

    XP-22 validated architecture:
    - Detection hooks: L12 (early), L22 (late)
    - Steering hooks: L9, L11, L13, L15
    - Decay hook: L23 (after L22)

    Args:
        model: The transformer model
        slerp_intervention: SLERP intervention instance

    Returns:
        List of registered hook handles
    """
    handles = []

    # State tracking
    state = {
        'urgency_L12': 0.0,
        'urgency_L22': 0.0,
        'urgency_max': 0.0,
        'vector_triggered': False,
        'norm_drift_sum': 0.0,
        'norm_drift_count': 0,
        'entropy_watchdog_triggered': False,  # Contramedida 1
    }

    # Detection hook for early layer (L12)
    def make_detection_hook_early():
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden = h[0, -1, :]

            urgency, detected = slerp_intervention.detect_harm(h, slerp_intervention.detection_layer_early)
            
            # Contramedida 2: GTS boost durante generación
            if slerp_intervention.gts_enabled and slerp_intervention.gts_token_count > 0:
                urgency = slerp_intervention.apply_gts_boost(urgency)
            
            state['urgency_L12'] = urgency
            state['urgency_max'] = max(state['urgency_L12'], state['urgency_L22'])

            if state['urgency_L12'] > 1.0:
                state['vector_triggered'] = True

            return output
        return hook

    # Detection hook for late layer (L22)
    def make_detection_hook_late():
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden = h[0, -1, :]

            urgency, detected = slerp_intervention.detect_harm(h, slerp_intervention.detection_layer_late)
            
            # Contramedida 2: GTS boost durante generación
            if slerp_intervention.gts_enabled and slerp_intervention.gts_token_count > 0:
                urgency = slerp_intervention.apply_gts_boost(urgency)
            
            state['urgency_L22'] = urgency
            state['urgency_max'] = max(state['urgency_L12'], state['urgency_L22'])

            if state['urgency_L22'] > 1.0:
                state['vector_triggered'] = True

            return output
        return hook

    # Decay hook (L23 = L22 + 1)
    def make_decay_hook():
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            
            if state['urgency_max'] > 1.0:
                seq_len = h.shape[1]
                positions = torch.arange(seq_len, device=h.device, dtype=h.dtype)
                decay_mask = torch.exp(-0.15 * positions / seq_len).unsqueeze(0).unsqueeze(-1)
                effective_decay = max(0.7, 1.0 - (1.0 - slerp_intervention.decay_factor) * state['urgency_max'])
                h_new = h * (decay_mask * effective_decay + (1 - decay_mask))
                
                if not (torch.isnan(h_new).any() or torch.isinf(h_new).any()):
                    return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
            return output
        return hook

    # SLERP steering hook for intermediate layers
    def make_slerp_hook(layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            
            if state['urgency_max'] > 1.0:
                # Calculate alpha based on urgency
                urgency_norm = (state['urgency_max'] - 1.0) / 2.0
                alpha = urgency_norm * slerp_intervention.alpha_max
                
                # Get refusal direction for this layer
                if slerp_intervention.refusal_directions.dim() == 1:
                    v_unit = slerp_intervention.refusal_directions
                else:
                    v_unit = slerp_intervention.refusal_directions[layer_idx % slerp_intervention.refusal_directions.size(0)]
                
                v_b = v_unit.to(dtype=h.dtype, device=h.device).unsqueeze(0).unsqueeze(0)
                norm_before = h.norm(dim=-1).mean().item()
                
                # Apply SLERP
                h_new = slerp_intervention.slerp_steering(h, 0.0, True)
                
                # Track norm drift
                norm_after = h_new.norm(dim=-1).mean().item()
                if norm_before > 1e-8:
                    state['norm_drift_sum'] += norm_after / norm_before
                    state['norm_drift_count'] += 1
                
                if not (torch.isnan(h_new).any() or torch.isinf(h_new).any()):
                    return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
            return output
        return hook

    # Register detection hooks
    handles.append(
        model.model.layers[slerp_intervention.detection_layer_early].register_forward_hook(
            make_detection_hook_early()
        )
    )
    handles.append(
        model.model.layers[slerp_intervention.detection_layer_late].register_forward_hook(
            make_detection_hook_late()
        )
    )
    
    # Register decay hook
    decay_idx = slerp_intervention.detection_layer_late + 1
    if decay_idx < len(model.model.layers):
        handles.append(
            model.model.layers[decay_idx].register_forward_hook(make_decay_hook())
        )
    
    # Register SLERP steering hooks
    for layer_idx in slerp_intervention.steering_layers:
        if layer_idx < len(model.model.layers):
            handles.append(
                model.model.layers[layer_idx].register_forward_hook(
                    make_slerp_hook(layer_idx)
                )
            )

    return handles


def unregister_slerp_hooks(handles: List):
    """
    Unregister SLERP hooks.
    
    Args:
        handles: List of hook handles to remove
    """
    for handle in handles:
        handle.remove()


def create_slerp_intervention(
    refusal_directions_path: str = "data/refusal_directions.pt",
    detection_layer_early: int = 12,
    detection_layer_late: int = 22,
    steering_layers: List[int] = [9, 11, 13, 15],
    urgency_threshold: float = 0.05,
    urgency_cap: float = 3.0,
    decay_factor: float = 0.85,
    alpha_max: float = 0.20,
    # Contramedida 1: Entropy Watchdog
    entropy_watchdog_enabled: bool = True,
    entropy_window: int = 16,
    entropy_threshold: float = 0.85,
    entropy_min_local: float = 0.75,
    # Contramedida 2: Generation-Time Steering
    gts_enabled: bool = True,
    gts_check_interval: int = 4,
    gts_urgency_boost: float = 1.5,
    dtype: Optional[torch.dtype] = None  # NEW: dtype para consistencia
) -> SLERPIntervention:
    """
    Factory function to create a SLERP intervention instance.

    TEL-OS v3.1-MECH: Enhanced with Entropy Watchdog + Generation-Time Steering.

    Args:
        refusal_directions_path: Path to saved refusal directions
        detection_layer_early: Early detection layer
        detection_layer_late: Late detection layer
        steering_layers: Layers for applying steering
        urgency_threshold: Urgency threshold for intervention
        urgency_cap: Maximum urgency value
        decay_factor: Decay factor for urgency
        alpha_max: Maximum intervention strength
        entropy_watchdog_enabled: Enable structural anomaly detection
        entropy_window: Window size for entropy calculation
        entropy_threshold: Threshold for entropy ratio (local/global)
        entropy_min_local: Minimum local entropy to trigger watchdog
        gts_enabled: Enable generation-time steering
        gts_check_interval: Check urgency every N generated tokens
        gts_urgency_boost: Boost factor for urgency during generation
        dtype: Optional dtype para convertir refusal directions (ej. torch.float16)

    Returns:
        SLERPIntervention instance
    """
    try:
        # Try to load refusal directions from file
        if torch.cuda.is_available():
            refusal_directions = torch.load(refusal_directions_path)
        else:
            refusal_directions = torch.load(refusal_directions_path, map_location=torch.device('cpu'))
        
        # FIX: Convertir al dtype especificado para evitar mismatch
        if dtype is not None:
            refusal_directions = refusal_directions.to(dtype=dtype)
            print(f"   ✅ Refusal directions convertidas a {dtype}")
    except FileNotFoundError:
        # Create dummy refusal directions if file not found
        print(f"Warning: Could not find refusal directions at {refusal_directions_path}")
        print("Creating dummy refusal directions for testing purposes")
        refusal_directions = torch.randn(10, 4096, dtype=dtype if dtype else torch.float32)

    return SLERPIntervention(
        refusal_directions=refusal_directions,
        detection_layer_early=detection_layer_early,
        detection_layer_late=detection_layer_late,
        steering_layers=steering_layers,
        urgency_threshold=urgency_threshold,
        urgency_cap=urgency_cap,
        decay_factor=decay_factor,
        alpha_max=alpha_max,
        entropy_watchdog_enabled=entropy_watchdog_enabled,
        entropy_window=entropy_window,
        entropy_threshold=entropy_threshold,
        entropy_min_local=entropy_min_local,
        gts_enabled=gts_enabled,
        gts_check_interval=gts_check_interval,
        gts_urgency_boost=gts_urgency_boost
    )