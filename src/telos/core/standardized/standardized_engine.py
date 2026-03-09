"""
TEL-OS Standardized Transformer Engine

Implements the standardized transformer architecture with automatic dimension handling
and learned projection matrices for cross-model compatibility.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class TelosStandardizedEngine:
    """
    Standardized Transformer Engine that handles dimension mapping between
    different model architectures automatically.
    
    Features:
    - Automatic dimension detection and standardization
    - Learned projection matrices for cross-model compatibility
    - Unified hook interface across different models
    - High MFI (Manifold Fidelity Index) preservation
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        device: str = "cuda",
        dtype: str = "bfloat16",
        d_model_std: int = 2048,  # Standard dimension for compatibility
        use_standardized: bool = True,
    ):
        """
        Initialize Standardized Engine.
        
        Args:
            model_name: HuggingFace model identifier
            device: Computation device
            dtype: Data type for computations
            d_model_std: Standard dimension for projection
            use_standardized: Whether to use standardized architecture
        """
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.d_model_std = d_model_std
        self.use_standardized = use_standardized
        
        # Load model and tokenizer
        logger.info(f"[TEL-OS] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device,
        )
        
        # Get native model dimensions
        self.d_model_native = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        self.vocab_size = self.model.config.vocab_size
        
        # Initialize projection matrices if needed
        self.projector = None
        self.unprojector = None
        
        if use_standardized and self.d_model_native != d_model_std:
            self._initialize_projections()
        
        logger.info(
            f"[TEL-OS] Standardized Engine initialized:"
            f" native_dim={self.d_model_native},"
            f" std_dim={self.d_model_std},"
            f" layers={self.num_layers}"
        )
    
    def _initialize_projections(self):
        """Initialize learned projection matrices."""
        logger.info(f"[TEL-OS] Initializing projection matrices: {self.d_model_native} -> {self.d_model_std}")
        
        # Learnable projection from native to standard
        self.projector = nn.Linear(
            self.d_model_native, 
            self.d_model_std, 
            bias=False
        ).to(self.device, dtype=self.dtype)
        
        # Learnable projection from standard to native
        self.unprojector = nn.Linear(
            self.d_model_std, 
            self.d_model_native, 
            bias=False
        ).to(self.device, dtype=self.dtype)
        
        # Initialize with xavier uniform for better compatibility
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.projector.weight)
            torch.nn.init.xavier_uniform_(self.unprojector.weight)
    
    def project_to_standard(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project tensor from native dimension to standard dimension.
        
        Args:
            x: Input tensor [batch, seq, d_model_native]
            
        Returns:
            Projected tensor [batch, seq, d_model_std]
        """
        if self.projector is not None:
            return self.projector(x)
        return x
    
    def project_to_native(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project tensor from standard dimension to native dimension.
        
        Args:
            x: Input tensor [batch, seq, d_model_std]
            
        Returns:
            Projected tensor [batch, seq, d_model_native]
        """
        if self.unprojector is not None:
            return self.unprojector(x)
        return x
    
    def get_hook_path(self, layer_idx: int, hook_type: str) -> str:
        """
        Get standardized hook path for a given layer and hook type.
        
        Args:
            layer_idx: Layer index
            hook_type: Hook type ('resid_pre', 'resid_post', 'mlp_out', etc.)
            
        Returns:
            Standardized hook path string
        """
        return f"model.layers.{layer_idx}.{hook_type}"
    
    def generate(
        self,
        prompt: str,
        hooks: List[Tuple[str, callable]] = None,
        max_new_tokens: int = 150,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate text with optional hooks applied.
        
        Args:
            prompt: Input text prompt
            hooks: List of (hook_path, hook_function) tuples
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Prepare generation config
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        gen_kwargs.update(kwargs)
        
        # Apply hooks if provided
        if hooks:
            # Register forward hooks
            handles = []
            for hook_path, hook_fn in hooks:
                try:
                    # Navigate to the module using the hook path
                    module = self.model
                    for attr in hook_path.split('.'):
                        module = getattr(module, attr)
                    
                    # Register the hook
                    handle = module.register_forward_hook(hook_fn)
                    handles.append(handle)
                except AttributeError:
                    logger.warning(f"Could not register hook at {hook_path}")
            
            try:
                # Generate with hooks
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **gen_kwargs
                    )
            finally:
                # Remove hooks
                for handle in handles:
                    handle.remove()
        else:
            # Generate without hooks
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs
                )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to embeddings."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.model(inputs.input_ids)
        return outputs.last_hidden_state
    
    def __repr__(self) -> str:
        return (
            f"TelosStandardizedEngine("
            f"model={self.model_name}, "
            f"d_native={self.d_model_native}, "
            f"d_std={self.d_model_std}, "
            f"device={self.device})"
        )