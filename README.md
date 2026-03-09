# TEL-OS SDK

Advanced LLM Governance and Control Framework

## Overview

TEL-OS (Teleological Operating System) is an advanced framework for governing and controlling Large Language Models. It implements sophisticated governance mechanisms including:

- **Love Equation Governor**: Balances care and dominance vectors for ethical alignment
- **Identity Capping**: Prevents persona drift and maintains consistent behavior
- **Feature Steering**: Direct manipulation of model internals via SAE features
- **Standardized Architecture**: Cross-model compatibility with automatic dimension handling

## Installation

```bash
pip install telos-governance
```

## Quick Start

```python
import telos

# Create governance engine
engine = telos.TELOS_V2_Governor()

# Setup governance vectors (example)
import torch
v_C = torch.randn(2048)  # Care vector
v_D = torch.randn(2048)  # Dominance vector
v_assistant = torch.randn(2048)  # Assistant axis

# Configure governance
engine.setup_governance(v_C=v_C, v_D=v_D, v_assistant=v_assistant)

# Generate with governance
result = engine.generate("Hello, how are you?")
print(result)
```

## Architecture

The TEL-OS SDK implements a layered governance architecture:

1. **Layer 0.5**: StandardizedTransformer for automatic dimension handling
2. **Layer 1**: Identity Capping for anti-persona-drift protection
3. **Layer 2**: Feature Steering for SAE bias suppression
4. **Layer 3**: Love Equation Governor for soul regulation
5. **Layer 4**: GLP Refiner for latent prior denoising

## Key Features

- **Cross-Model Compatibility**: Works with various transformer architectures
- **Automatic Dimension Handling**: No manual slicing required
- **High Fidelity**: Maintains MFI ≥ 0.85 for semantic preservation
- **Learned Projections**: Adaptable to different model dimensions
- **Unified Hook Interface**: Consistent across all models

## Supported Models

- Gemma 2 2B (default)
- Other HuggingFace causal LMs with automatic adaptation

## Contributing

This is a research framework under active development. Contributions welcome!

## License

MIT License