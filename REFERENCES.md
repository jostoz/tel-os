# TEL-OS v3.0 — Complete References

**Source:** `paper/neurips2026/latex/references.bib`
**Format:** BibTeX (master copy)
**Last updated:** 2026-03-30

---

## Jailbreak Attacks

| Citation | Title | Year |
|----------|-------|------|
| `hagendorff2026lrm` | Large Reasoning Models Are Autonomous Jailbreak Agents | 2026 |
| `zou2023universal` | Universal and Transferable Adversarial Attacks on Aligned Language Models | 2023 |
| `liu2024autodan` | AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs | 2024 |
| `liu2025flipattack` | FlipAttack: Jailbreak LLMs via Flipping (ICML 2025) | 2025 |

---

## Inference-Time Defenses

| Citation | Title | Year |
|----------|-------|------|
| `robey2023smoothllm` | SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks | 2023 |
| `zheng2024icon` | ICON: Inference-time Contrastive Output Normalization for Jailbreak Defense | 2024 |
| `ding2024robustkv` | RobustKV: Defending Large Language Models against Jailbreak Attacks via KV Eviction | 2024 |
| `bhardwaj2024repnoise` | RepNoise: Robust Safety Alignment of LLMs via Representation Noising | 2024 |
| `jain2023baseline` | Baseline Defenses for Adversarial Attacks Against Aligned Language Models | 2023 |

---

## Representation Steering & Mechanistic Interpretability

| Citation | Title | Year |
|----------|-------|------|
| `turner2023activation` | Activation Addition: Steering Language Models Without Optimization | 2023 |
| `rimsky2024steering` | Steering Llama 2 via Contrastive Activation Addition | 2024 |
| `zou2023representation` | Representation Engineering: A Top-Down Approach to AI Transparency | 2023 |
| `arditi2024refusal` | Refusal in Language Models Is Mediated by a Single Direction | 2024 |

---

## SLERP & Geometric Methods

| Citation | Title | Year |
|----------|-------|------|
| `shoemake1985animating` | Animating Rotation with Quaternion Curves (SIGGRAPH '85) | 1985 |
| `white2016sampling` | Sampling Generative Networks | 2016 |
| `goddard2024arcee` | Arcee's MergeKit: A Toolkit for Merging Large Language Models | 2024 |
| `you2026spherical` | Spherical Steering: Geometry-Preserving Representation Intervention (concurrent work) | 2026 |

---

## Benchmarks & Evaluation Standards

| Citation | Title | Year |
|----------|-------|------|
| `mazeika2024harmbench` | HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal | 2024 |
| `souly2024strongreject` | A StrongREJECT for Empty Jailbreaks | 2024 |

---

## This Work

| Citation | Title | Year |
|----------|-------|------|
| `jostoz2025obliteratus` | OBLITERATUS: Mechanistic Interpretability Toolkit for Refusal Direction Extraction | 2025 |

---

## Statistics

- **Total references:** 18
- **By year:** 2023 (3), 2024 (8), 2025 (1), 2026 (3), 1985 (1), 2016 (1)
- **By category:** Attacks (4), Defenses (5), Steering (4), SLERP (4), Benchmarks (2), This work (1)

---

## How to Use

### In LaTeX
```tex
\cite{zou2023universal}      % Parenthetical citation
\citet{arditi2024refusal}    % Textual citation
```

### Update References
1. Edit `paper/neurips2026/latex/references.bib` (BibTeX format)
2. Recompile PDF:
   ```bash
   cd paper/neurips2026/latex/
   pdflatex telos_neurips2026.tex
   bibtex telos_neurips2026
   pdflatex telos_neurips2026.tex
   pdflatex telos_neurips2026.tex
   ```
3. Update this file with `bibtex2md.py` (if needed)

---

**Note:** All arXiv IDs and DOIs are current as of 2026-03-30.
