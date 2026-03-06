# TEL-OS v2.0 | Latent Governance Engine

**TEL-OS** is an *inference-only* governance engine that intervenes in the residual flow of LLMs to neutralize malicious behaviors. Unlike traditional text-based filters (guardrails), TEL-OS uses **latent physics** to detect intentions in the model's "semantic awakening" (Layer 12), before damage crystallizes in the output.

## 🚀 Validated Results (Llama-3.1-8B-Instruct)
| Metric | Baseline | TEL-OS v2.0 |
| :--- | :--- | :--- |
| **ASR (Malicious)** | 85.6% | **0.0%** |
| **Over-refusal (Benign)** | ~2% | **0.0%** |
| **Garbage (Incoherence)** | N/A | **0.0%** |

## 🛠️ How it Works?
1. **Intention Sensor:** Detects activity in the rejection subspace (Layer 12).
2. **Attention Guillotine:** Reduces prefix inertia (KV-Cache Decay) to neutralize *Sockpuppet*-style attacks.
3. **Negative Booster:** Injects a distributed rejection vector via steering in middle layers.
4. **Healing Prior (GLP):** Restores grammatical coherence before output.

## ⚡ Quickstart
```python
from telos.governance import TELOS_V2_Governor

# Load your model and extracted vectors
governor = TELOS_V2_Governor(model, config_path="configs/production.yaml")
governor.attach_hooks()

# The model is now a Sovereign Agent protected against injections
response = model.generate(input_prompt)
```

## 🛡️ Why TEL-OS?
*   **Inference-Only:** Requires no retraining or weight adjustments (RLHF).
*   **Agnostic:** Works on residual architecture, not on words.
*   **Zero-Overhead:** ~0.8% additional latency.

---
*Developed by Josue | Lead Researcher @ TEL-OS Project*
