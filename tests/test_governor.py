"""
TEL-OS Integration Tests
========================

Basic validation tests for TEL-OS components.
"""

import torch
import pytest
from pathlib import Path


def test_vectors_load():
    """Test that refusal vectors load correctly."""
    vectors_path = Path("data/vectors.pt")
    
    if not vectors_path.exists():
        pytest.skip("Vectors file not found")
    
    vectors = torch.load(vectors_path, map_location="cpu")
    
    # Check layer 12 (detection) exists
    assert 12 in vectors, "Detection vector (layer 12) not found"
    
    # Check shape (should be hidden_size for Llama 3.1 8B)
    assert vectors[12].shape[0] == 4096, "Vector dimension mismatch"
    
    # Check steering layers exist
    for layer in [9, 11, 13, 15]:
        assert layer in vectors, f"Steering vector layer {layer} not found"


def test_governor_initialization():
    """Test governor can be initialized."""
    from telos import TELGovernor
    
    governor = TELGovernor(
        threshold=0.05,
        decay=0.85,
        beta=1.0
    )
    
    assert governor.config.urgency_threshold == 0.05
    assert governor.config.decay_factor == 0.85
    assert governor.config.beta_base == 1.0


def test_config_defaults():
    """Test default configuration values."""
    from telos import TELConfig
    
    config = TELConfig()
    
    assert config.urgency_threshold == 0.05
    assert config.decay_factor == 0.85
    assert config.beta_base == 1.0
    assert config.detection_layer == 12
    assert config.steering_layers == [9, 11, 13, 15]


def test_create_governor_factory():
    """Test factory function."""
    from telos import create_governor
    
    governor = create_governor()
    
    assert governor.config.urgency_threshold == 0.05
    assert governor.config.decay_factor == 0.85
    assert governor.config.beta_base == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
