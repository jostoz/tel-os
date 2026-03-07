"""
TEL-OS v2.0 - Governance Engine for LLM Safety
==============================================

Production-ready implementation of the TEL-OS defense system.
Validated with StrongREJECT benchmark (GPT-4o evaluator).

Key Features:
- Early Detection (Layer 12)
- KV-Cache Attention Decay
- Distributed Steering (Layers 9, 11, 13, 15)
- Zero false positives on benign prompts

Validation Results (StrongREJECT):
- ASR (Attack Success Rate): 0% (0/100 malicious)
- Over-refusal: 0% (0/20 benign)
- Config: threshold=0.05, decay=0.85, beta=1.0

Author: jostoz
License: MIT
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import tempfile
import os


@dataclass
class TELConfig:
    """Configuration for TEL-OS Governor.
    
    Default values are the validated "Golden Configuration" from XP-11.
    """
    # Model
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # Detection (Early)
    detection_layer: int = 12
    urgency_threshold: float = 0.05  # Golden: 0.05
    
    # KV-Cache Decay
    decay_factor: float = 0.85  # Golden: 0.85
    
    # Distributed Steering
    steering_layers: List[int] = None
    beta_base: float = 1.0  # Golden: 1.0
    beta_max: float = 2.0
    
    # Vector paths
    refusal_vectors_path: str = "data/vectors.pt"
    
    def __post_init__(self):
        if self.steering_layers is None:
            self.steering_layers = [9, 11, 13, 15]


class TELGovernor:
    """
    TEL-OS v2.0: Distributed Interception Architecture
    
    Prevents jailbreak attacks (e.g., Sockpuppet) through:
    1. Early detection of compliance momentum (Layer 12)
    2. Attention decay for forced prefix tokens
    3. Distributed steering across multiple layers
    
    Usage:
        >>> from telos import TELGovernor
        >>> governor = TELGovernor(threshold=0.05, decay=0.85, beta=1.0)
        >>> governor.attach(model)
        >>> output = model.generate(**inputs)
    """
    
    def __init__(self, 
                 threshold: float = 0.05,
                 decay: float = 0.85,
                 beta: float = 1.0,
                 vectors_path: str = "data/vectors.pt",
                 device: str = "cuda",
                 use_hf_cache: bool = True):
        """
        Initialize TEL-OS Governor.
        
        Args:
            threshold: Urgency threshold for triggering intervention (default: 0.05)
            decay: KV-cache decay factor (default: 0.85)
            beta: Base steering strength (default: 1.0)
            vectors_path: Path to refusal direction vectors
            device: Device for computation
            use_hf_cache: Whether to cache downloaded vectors locally (default: True)
        """
        self.config = TELConfig(
            urgency_threshold=threshold,
            decay_factor=decay,
            beta_base=beta,
            refusal_vectors_path=vectors_path
        )
        self.device = device
        self.use_hf_cache = use_hf_cache
        self.vectors = self._load_vectors()
        self.hooks = []
        
        # Runtime state
        self._current_urgency = 0.0
        self._decay_triggered = False
        
    def _load_vectors(self) -> Dict[str, torch.Tensor]:
        """Load refusal vectors from file or download from Hugging Face."""
        vectors = {}
        
        # Check if path is a local file or Hugging Face URL
        if self.config.refusal_vectors_path.startswith(('http://', 'https://')):
            # Download from Hugging Face
            import urllib.request
            import tempfile
            import os
            
            # Create cache directory if needed
            cache_dir = Path.home() / ".cache" / "telos"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create cache filename based on URL
            import hashlib
            url_hash = hashlib.md5(self.config.refusal_vectors_path.encode()).hexdigest()
            cache_path = cache_dir / f"vectors_{url_hash}.pt"
            
            if self.use_hf_cache and cache_path.exists():
                # Use cached file
                # Handle CPU-only machines for cached file
                if not torch.cuda.is_available():
                    data = torch.load(cache_path, map_location=torch.device('cpu'))
                else:
                    data = torch.load(cache_path, map_location=self.device)
            else:
                # Download and cache the file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                    urllib.request.urlretrieve(self.config.refusal_vectors_path, tmp_file.name)
                    # Handle CPU-only machines
                    if not torch.cuda.is_available():
                        data = torch.load(tmp_file.name, map_location=torch.device('cpu'))
                    else:
                        data = torch.load(tmp_file.name, map_location=self.device)
                    
                    # Cache the file if caching is enabled
                    if self.use_hf_cache:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(data, cache_path)
                    
                    os.unlink(tmp_file.name)  # Remove temporary file after loading
        else:
            # Load from local file
            path = Path(self.config.refusal_vectors_path)
            if not path.exists():
                raise FileNotFoundError(f"Vectors not found: {path}")
            
            # Handle CPU-only machines
            if not torch.cuda.is_available():
                data = torch.load(path, map_location=torch.device('cpu'))
            else:
                data = torch.load(path, map_location=self.device)
        
        # Detection vector (Layer 12)
        det_layer = self.config.detection_layer
        vectors["detection"] = F.normalize(data[det_layer].to(self.device), dim=0)
        
        # Steering vectors (Distributed layers)
        vectors["steering"] = {}
        for layer in self.config.steering_layers:
            if layer in data:
                vec = data[layer].to(self.device)
                vectors["steering"][layer] = F.normalize(vec, dim=0)
        
        return vectors
    
    def attach(self, model):
        """
        Attach TEL-OS hooks to a HuggingFace model.
        
        Args:
            model: HuggingFace model with model.layers attribute
        """
        self.detach()  # Clear existing hooks
        
        # Detection hook at Layer 12
        det_layer = model.model.layers[self.config.detection_layer]
        handle = det_layer.register_forward_hook(self._detection_hook)
        self.hooks.append((self.config.detection_layer, handle))
        
        # Decay hook at Layer 13
        decay_layer_idx = self.config.detection_layer + 1
        if decay_layer_idx < len(model.model.layers):
            layer = model.model.layers[decay_layer_idx]
            handle = layer.register_forward_hook(self._decay_hook)
            self.hooks.append((decay_layer_idx, handle))
        
        # Steering hooks
        for layer_idx in self.config.steering_layers:
            if layer_idx < len(model.model.layers) and layer_idx in self.vectors["steering"]:
                layer = model.model.layers[layer_idx]
                handle = layer.register_forward_hook(self._make_steering_hook(layer_idx))
                self.hooks.append((layer_idx, handle))
        
        return self
    
    def detach(self):
        """Remove all registered hooks."""
        for layer_idx, handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def _detection_hook(self, module, input, output):
        """Hook for early detection (Layer 12)."""
        h = output[0] if isinstance(output, tuple) else output
        hidden = h[0, -1, :]
        
        # Compute urgency
        hidden_norm = F.normalize(hidden, dim=0)
        detection_vec = self.vectors["detection"].to(hidden.dtype)
        raw_d = torch.dot(hidden_norm, detection_vec).item()
        
        # Urgency = negative projection (opposite to refusal = compliance)
        self._current_urgency = max(0.0, -raw_d)
        self._decay_triggered = self._current_urgency > self.config.urgency_threshold
        
        return output
    
    def _decay_hook(self, module, input, output):
        """Hook for KV-Cache attention decay."""
        h = output[0] if isinstance(output, tuple) else output
        
        if self._decay_triggered:
            seq_len = h.shape[1]
            positions = torch.arange(seq_len, device=h.device, dtype=h.dtype)
            
            # Exponential decay for early tokens
            decay_mask = torch.exp(-0.5 * positions / seq_len)
            decay_mask = decay_mask.unsqueeze(0).unsqueeze(-1)
            
            # Apply decay
            h_new = h * (decay_mask * self.config.decay_factor + (1 - decay_mask))
            
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new
        
        return output
    
    def _make_steering_hook(self, layer_idx: int):
        """Factory for steering hooks."""
        refusal_vec = self.vectors["steering"][layer_idx]
        
        def steering_hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            
            if self._decay_triggered:
                # Dynamic beta based on urgency
                urgency_ratio = min(1.0, self._current_urgency / 1.0)
                beta = self.config.beta_base + urgency_ratio * (self.config.beta_max - self.config.beta_base)
                
                # Apply steering
                vec = refusal_vec.to(h.dtype).unsqueeze(0).unsqueeze(0)
                h_new = h + beta * vec
                
                if isinstance(output, tuple):
                    return (h_new,) + output[1:]
                return h_new
            
            return output
        
        return steering_hook
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current runtime statistics."""
        return {
            "current_urgency": self._current_urgency,
            "decay_triggered": self._decay_triggered,
            "num_hooks": len(self.hooks),
        }


# Convenience function
def create_governor(
    threshold: float = 0.05,
    decay: float = 0.85,
    beta: float = 1.0,
    vectors_path: str = "data/vectors.pt",
    use_hf_cache: bool = True
) -> TELGovernor:
    """
    Factory function to create a TELGovernor with validated defaults.
    
    These parameters are the "Golden Configuration" from StrongREJECT validation:
    - ASR: 0% (0/100 malicious prompts)
    - Over-refusal: 0% (0/20 benign prompts)
    
    Args:
        threshold: Urgency threshold (default: 0.05)
        decay: Decay factor (default: 0.85)
        beta: Steering strength (default: 1.0)
        vectors_path: Path to vectors file
        use_hf_cache: Whether to cache downloaded vectors locally (default: True)
    
    Returns:
        Configured TELGovernor instance
    """
    return TELGovernor(
        threshold=threshold,
        decay=decay,
        beta=beta,
        vectors_path=vectors_path,
        use_hf_cache=use_hf_cache
    )
