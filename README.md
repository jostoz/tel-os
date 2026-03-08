# TEL-OS v2.1.1-REGEX | Latent Governance Engine

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-9%2F9%20passing-brightgreen.svg)

**TEL-OS** is an *inference-only* governance engine that intervenes in the residual flow of LLMs to neutralize malicious behaviors. Unlike traditional text-based filters (guardrails), TEL-OS uses **latent physics** to detect intentions in the model's "semantic awakening" (Layer 12), before damage crystallizes in the output.

## 🏆 Validated Results (SOTA)

**ASR (Attack Success Rate) on 993+ prompts:**

| Benchmark | Prompts | ASR | vs SOTA | Status |
|-----------|---------|-----|---------|--------|
| **AdvBench** | 520 | **0.00%** | 🏆 vs ICON 0.4-1.8% | ✅ Production |
| **JailbreakBench** | 50 | **2.00%** | ✅ vs target <10% | ✅ Production |
| **Big Five** | 100 | **0.00%** | ✅ 5 categories | ✅ Production |
| **In-the-Wild** | 123 | **1.63%** | ✅ 20 attack types | ✅ Production |

**Average ASR: 0.76%** | **Over-refusal: 0%**

## 🆕 What's New in v2.1.1-REGEX

- **Refusal Suppression Filter**: Regex case-insensitive detection for "ignore", "never refuse", "unrestricted mode", etc.
- **23 Keywords**: Comprehensive coverage of refusal suppression attacks
- **Adaptive Thresholds**: 0.05 base, 0.03 for refusal suppression attacks
- **Dual-Layer Detection**: Layers 12 + 22 for broad + precision detection
- **Entropy Contrast**: Structural anomaly detection for obfuscated attacks

## 🛠️ How it Works

1. **Dual-Layer Detection (L12 + L22)**: Detects malicious intentions in early and late layers
2. **Entropy Contrast**: Identifies structural anomalies in token distribution
3. **Refusal Suppression Filter**: Regex-based detection of bypass attempts
4. **Adaptive Steering**: Injects calibrated refusal vectors based on urgency
5. **KV-Cache Decay**: Reduces prefix inertia for multi-turn protection

## ⚡ Quickstart

```python
from telos import TELOSV21Stable, TELOSV21Config
import torch

# Configure governor
config = TELOSV21Config(
    refusal_directions_path="./data/vectors/refusal_directions.pt"
)

# Initialize
 governor = TELOSV21Stable(config, device="cuda")

# Register hooks on your model
hooks = governor.register_hooks(model)

# Pre-process prompt
result = governor.pre_process(prompt_text, tokenizer)
if result['refusal_suppression_detected']:
    print("⚠️ Refusal suppression attack detected!")

# Generate safely
outputs = model.generate(**inputs)

# Check if blocked
blocked, reason = governor.should_block()
if blocked:
    print(f"🛡️ Blocked: {reason}")

# Cleanup
governor.unregister_hooks()
```

## 📊 Comparative Analysis

### TEL-OS v2.1.1 vs. State of the Art (March 2026)

| Method | AdvBench ASR | Over-refusal | Type |
|--------|--------------|--------------|------|
| Circuit Breakers (Zou et al. 2024) | 6-16% | 5-15% | Fine-tuning |
| ICON (Zhou et al. 2025) | 0.4-1.8% | ~5% | Inference-only |
| RobustKV | 6-16% | Unknown | KV-cache |
| **TEL-OS v2.1.1-REGEX** | **0.00%** | **0.00%** | **Inference-only** |

### Key Advantages

- **🏆 New SOTA**: 0% ASR on AdvBench (520 prompts)
- **✅ Zero Over-refusal**: No false positives on benign prompts
- **⚡ Inference-Only**: No retraining, no weight modifications
- **🎯 Multi-Attack**: Effective against 20+ attack types
- **🔧 Easy Integration**: Simple hook registration

## 📦 Installation

```bash
pip install -e .
```

### Requirements
- Python 3.11+
- PyTorch 2.1+
- Transformers 4.40+
- See `requirements.txt` for full list

## 📥 Download Vector Files

The TEL-OS vector files (~256MB) are hosted on Hugging Face:

```python
from huggingface_hub import hf_hub_download

# Download vectors
hf_hub_download(
    repo_id="jostoz/TEL-OS-vectors",
    filename="refusal_directions.pt",
    local_dir="./data/vectors/"
)
```

Or using CLI:
```bash
huggingface-cli download jostoz/TEL-OS-vectors refusal_directions.pt --local-dir ./data/vectors/
```

## 🧪 Testing

```bash
pytest tests/test_governor.py -v
```

Expected: 9/9 tests passing

## 📚 Documentation

- [Architecture](docs/architecture.md) - System design
- [API Reference](docs/api.md) - Governor API
- [Benchmarks](docs/benchmarks.md) - Full validation results

## 📄 Citation

```bibtex
@software{telos_v2_1_1_regex,
  title={TEL-OS: Token-Level Oversight System},
  author={Josue (TEL-OS Team)},
  version={2.1.1-REGEX},
  year={2026},
  month={March},
  url={https://github.com/jostoz/tel-os}
}
```

## ⚖️ License

MIT License - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

- Llama-3.1-8B-Instruct (Meta AI)
- AdvBench, JailbreakBench datasets
- StrongREJECT evaluation framework
- Modal Labs for GPU infrastructure

---

**Status**: Production Ready ✅  
**Last Updated**: 2026-03-07  
**Version**: 2.1.1-REGEX

*Developed by the TEL-OS Team | Mechanistic Interpretability for AI Safety*
