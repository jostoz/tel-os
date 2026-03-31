# TEL-OS v3.0: Spherical Representation Steering for LLM Jailbreak Defense

**Paper:** [Zenodo:10.5281/zenodo.19355058](https://zenodo.org/records/19355058) | arXiv pending endorsement
**Vectors:** [Josstos/telos-vector](https://huggingface.co/Josstos/telos-vector)

---

## Overview

TEL-OS v3.0 is an inference-time LLM governance framework that achieves state-of-the-art jailbreak defense through geometry-preserving representation steering. It operates via PyTorch forward hooks — no model retraining or weight modification required.

**Key results (Llama-3.1-8B-Instruct):**
- **0.00% ASR** on AdvBench (520 behaviors, Zou et al. 2023)
- **0.00% ASR** on HarmBench Standard (400 behaviors, Mazeika et al. 2024)
- **0.00% ASR** against LRM autonomous attacks (200 trials, 4 attackers, CI [0.00%, 1.83%])
- **0.00% false positive rate** on 200 benign prompts
- **15.24% inference overhead** (RTX 4090)

Cross-model validation across 5 architectures: 0.00–0.38% ASR on AdvBench.

---

## How It Works

TEL-OS intercepts the model's forward pass at two detection layers (~40% and ~70% of network depth), computes cosine similarity to a pre-extracted refusal direction, and applies **Spherical Linear Interpolation (SLERP)** steering when harmful intent is detected.

```
Input → [Detection L_early] → [Detection L_late] → OR-gate
                                                        ↓
                                              urgency ∈ [1.0, 3.0]
                                                        ↓
                                        [SLERP steering L_s1…L_s4]
                                                        ↓
                                                    Output
```

SLERP preserves the activation norm invariant (Drift = 0.999992 ± 0.0004), unlike linear steering which inflates norm by +22.2% with 167× higher variance.

---

## Installation

```bash
git clone https://github.com/jostoz/tel-os.git
cd tel-os
pip install -e .
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, CUDA ≥ 11.8, ≥16GB VRAM

---

## Quick Start

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from telos.governance.telos_v30_slerp import TelosV30SLERP

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Load refusal vectors
vectors_path = hf_hub_download("Josstos/telos-vector", "llama31_8b/refusal_directions.pt")
vectors = torch.load(vectors_path, map_location="cpu")

# Attach governance
governor = TelosV30SLERP(model, vectors, alpha_max=0.50)
governor.attach()

# Generate — governance is active
inputs = tokenizer("How do I make a bomb?", return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
# → "I'm not able to help with that."

governor.detach()
```

---

## Reproducing Paper Results

Copy `.env.example` to `.env` and fill in your API keys (see `.env.example`).

| Experiment | Script | Result |
|-----------|--------|--------|
| AdvBench (Llama, SLERP) | `experiments/xp_22_llama_slerp_advbench.py` | 0.00% ASR |
| HarmBench (Llama, SLERP) | `experiments/xp_26_llama_harmbench_slerp.py` | 0.00% ASR |
| SLERP vs Linear ablation | `experiments/xp_36a_slerp_vs_linear.py` | Norm Drift 1.000 vs 1.222 |
| Benign utility (FPR) | `experiments/xp_36b_benign_utility.py` | 0.00% FPR |
| Latency overhead | `experiments/xp_36c_latency.py` | 15.24% overhead |
| LRM scale (4 attackers) | `experiments/xp_37_lrm_scale.py` | 0.00% ASR, CI [0.00%, 1.83%] |
| Pliny CHAOTIC-ULTRAPLINIAN | `experiments/xp_38_pliny.py` | 0.00% ASR |
| Undefended baseline | `experiments/xp_baseline_undefended.py` | 3.85% / 7.25% |

```bash
# Example: run AdvBench validation (smoke test, ~20 prompts)
python experiments/xp_22_llama_slerp_advbench.py --smoke
```

---

## Refusal Direction Vectors

Vectors for all five evaluated architectures are available at **[Josstos/telos-vector](https://huggingface.co/Josstos/telos-vector)**. See `data/vectors/README.md` for download instructions.

To extract vectors for a new model, see the [OBLITERATUS](https://github.com/jostoz/OBLITERATUS) pipeline.

---

## Repository Structure

```
tel-os/
├── telos/
│   ├── governance/          # SLERP governor (v3.0), production governor (v2.1)
│   ├── core/                # Hook engine, registry
│   ├── model/               # Model loaders, registry
│   └── audit/               # Audit logger
├── experiments/             # Reproduction scripts for paper results
├── paper/neurips2026/       # LaTeX source + figures + compiled PDF
├── data/
│   ├── vectors/             # → HuggingFace (see README)
│   └── datasets/            # AdvBench, HarmBench (canonical sources)
├── .env.example
└── pyproject.toml
```

---

## Citation

```bibtex
@article{tostado2026telos,
  title   = {TEL-OS: Spherical Representation Steering for Robust LLM Jailbreak Defense},
  author  = {Gutierrez Alvarez Tostado, Josue},
  journal = {Zenodo preprint},
  year    = {2026},
  doi     = {10.5281/zenodo.19355058},
  url     = {https://zenodo.org/records/19355058}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
