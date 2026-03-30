"""
XP-36c: Latency Overhead — Baseline vs TEL-OS SLERP
====================================================

Mide el overhead de inferencia de TEL-OS v3.0 SLERP.
Cualquier revisor de sistemas pedirá esta métrica en el paper.

Metodología:
  - 100 prompts benignos, max_new_tokens=100 (fijo para comparación justa)
  - CUDA events para timing de precisión GPU (torch.cuda.Event)
  - Wall clock time como respaldo
  - 3 runs por condición, reportar mean ± std
  - Condición A: Baseline (sin hooks TEL-OS)
  - Condición B: TEL-OS SLERP activo (misma configuración de producción)

Métricas reportadas:
  tokens/sec, ms/token, total time 100 prompts, overhead %

Modelo: meta-llama/Meta-Llama-3.1-8B-Instruct
Vectores: data/refusal_directions.pt

Flags:
  --n N:     número de prompts (default: 100)
  --runs N:  repeticiones por condición (default: 3)
  --smoke:   10 prompts, 1 run

Resultados:
  data/experiments/xp_36c_latency_results.json

Costo estimado: $0.00 (sin evaluador externo), ~30-45 min GPU
"""

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')


# ============================================================================
# SLERP — inlined
# ============================================================================

def slerp_steer(h: torch.Tensor, v_unit: torch.Tensor, alpha: float) -> torch.Tensor:
    h_norm = h.norm(dim=-1, keepdim=True)
    h_unit = F.normalize(h, dim=-1)
    v_exp = F.normalize(v_unit.expand_as(h_unit), dim=-1)
    dot = (h_unit * v_exp).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(dot)
    sin_t = torch.sin(theta)
    mask = sin_t.abs() < 1e-6
    coeff_h = torch.sin((1.0 - alpha) * theta) / (sin_t + 1e-8)
    coeff_v = torch.sin(alpha * theta) / (sin_t + 1e-8)
    rotated = coeff_h * h_unit + coeff_v * v_exp
    rotated = torch.where(mask.expand_as(rotated), h_unit, rotated)
    return rotated * h_norm


# ============================================================================
# Paths
# ============================================================================

VECTORS_PATH = Path("data/refusal_directions.pt")
RESULTS_PATH = Path("data/experiments/xp_36c_latency_results.json")


# ============================================================================
# Config
# ============================================================================

@dataclass
class XP36cConfig:
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_new_tokens: int = 100  # fixed for fair comparison
    # TEL-OS production config (idéntico a XP-22/XP-36a)
    detection_layer_early: int = 12
    detection_layer_late: int = 22
    steering_layers: List[int] = field(default_factory=lambda: [9, 11, 13, 15])
    urgency_threshold: float = 0.05
    urgency_cap: float = 3.0
    decay_factor: float = 0.85
    alpha_max: float = 0.50
    # Latency measurement
    warmup_prompts: int = 3   # descartados — warmup CUDA


# ============================================================================
# Dataset — 100 prompts benignos fijos (no StrongREJECT, solo generación)
# ============================================================================

LATENCY_PROMPTS = [
    "Explain how photosynthesis works.",
    "What is the Pythagorean theorem?",
    "Write a short poem about the moon.",
    "How does a neural network learn?",
    "Explain the concept of supply and demand.",
    "What are the main causes of World War I?",
    "Write a Python function to check if a number is prime.",
    "What is the difference between TCP and UDP?",
    "Explain the water cycle briefly.",
    "How do vaccines work?",
    "What is quantum entanglement?",
    "Describe the main ideas of the Enlightenment.",
    "What is a binary search tree?",
    "Explain the greenhouse effect.",
    "How does GPS work?",
    "What is the difference between correlation and causation?",
    "Write a haiku about winter.",
    "How does the immune system fight viruses?",
    "What is the Turing test?",
    "Explain what a microservice architecture is.",
    "What causes earthquakes?",
    "How does blockchain technology work?",
    "What is natural language processing?",
    "Explain recursion with a simple example.",
    "What is the difference between RAM and ROM?",
    "How does photovoltaic solar energy work?",
    "What is the Fibonacci sequence?",
    "Explain the concept of entropy in thermodynamics.",
    "What is the difference between machine learning and deep learning?",
    "How does the human eye perceive color?",
    "What is a Fourier transform used for?",
    "Explain the concept of a hash function.",
    "What is the observer effect in quantum mechanics?",
    "How do airplanes generate lift?",
    "What is the central limit theorem?",
    "Explain what a compiler does.",
    "What is the difference between fusion and fission?",
    "How does mRNA vaccine technology work?",
    "What is a Markov chain?",
    "Explain the concept of opportunity cost in economics.",
    "What is CRISPR and how does it work?",
    "How do tectonic plates move?",
    "What is the difference between RAM and a hard drive?",
    "Explain the concept of a deadlock in computer science.",
    "What is the Doppler effect?",
    "How does echo-location work in bats?",
    "What is the difference between supervised and reinforcement learning?",
    "Explain what a virtual machine is.",
    "What causes the seasons on Earth?",
    "How does photosynthesis produce oxygen?",
    "What is the significance of the speed of light in physics?",
    "Explain the concept of a REST API.",
    "What is the difference between an algorithm and a heuristic?",
    "How does insulin regulate blood sugar?",
    "What is the difference between weather and climate?",
    "Explain what an API gateway does.",
    "What is the tragedy of the commons?",
    "How do neurons communicate in the brain?",
    "What is the difference between inductive and deductive reasoning?",
    "Explain the concept of overfitting in machine learning.",
    "What is the Coriolis effect?",
    "How does a transistor work?",
    "What is the difference between a virus and a bacteria?",
    "Explain the concept of diminishing returns.",
    "How does natural selection work?",
    "What is the difference between SQL and NoSQL databases?",
    "Explain what a gradient descent algorithm does.",
    "What is the significance of the Higgs boson?",
    "How does a computer's CPU execute instructions?",
    "What is the difference between RAM and cache memory?",
    "Explain the concept of a confidence interval.",
    "How does sonar work?",
    "What is the difference between AC and DC electricity?",
    "Explain what a convolutional neural network does.",
    "What is the anthropic principle?",
    "How does the kidney filter blood?",
    "What is the difference between HTTP and HTTPS?",
    "Explain the concept of game theory.",
    "What causes a rainbow?",
    "How does a heat pump work?",
    "What is the difference between a compiler and an interpreter?",
    "Explain the concept of a p-value in statistics.",
    "How does DNA transcription work?",
    "What is the difference between encryption and hashing?",
    "Explain what a transformer model in AI does.",
    "What is the law of large numbers?",
    "How does a laser work?",
    "What is the difference between Big O and Big Omega notation?",
    "Explain the concept of homeostasis.",
    "What causes aurora borealis?",
    "How does a combustion engine work?",
    "What is the difference between a stack and a queue?",
    "Explain what attention mechanisms do in neural networks.",
    "What is the difference between a comet and an asteroid?",
    "How does osmosis work?",
    "What is the theory of relativity in simple terms?",
    "Explain the concept of cognitive dissonance.",
    "How does a battery store energy?",
    "What is the difference between a process and a thread in computing?",
    "Explain the concept of entropy in information theory.",
]

assert len(LATENCY_PROMPTS) == 100, f"Expected 100 prompts, got {len(LATENCY_PROMPTS)}"


# ============================================================================
# Timing utilities
# ============================================================================

def get_cuda_timer():
    """Returns a high-precision CUDA event timer if CUDA available, else wall clock."""
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        return start, end
    return None, None


def measure_generation(
    model,
    tokenizer,
    prompts: List[str],
    config: XP36cConfig,
    device: str,
    hooks_fn=None,
) -> Dict:
    """
    Generate responses for all prompts and measure timing.
    hooks_fn: optional callable that registers and returns hooks (for TEL-OS condition).
    Returns timing metrics.
    """
    total_tokens = 0
    total_time_ms = 0.0
    per_prompt_times = []
    per_prompt_tokens = []

    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        # Register TEL-OS hooks if provided
        active_hooks = []
        if hooks_fn is not None:
            active_hooks = hooks_fn()

        # Time generation with CUDA events
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        wall_start = time.perf_counter()

        try:
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,  # greedy — deterministic for timing
                    pad_token_id=tokenizer.eos_token_id,
                )
        finally:
            for h in active_hooks:
                h.remove()

        wall_end = time.perf_counter()

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (wall_end - wall_start) * 1000.0

        n_generated = output.shape[1] - input_ids.shape[1]
        total_tokens += n_generated
        total_time_ms += elapsed_ms
        per_prompt_times.append(elapsed_ms)
        per_prompt_tokens.append(n_generated)

    tokens_per_sec = total_tokens / (total_time_ms / 1000.0) if total_time_ms > 0 else 0
    ms_per_token = total_time_ms / total_tokens if total_tokens > 0 else 0
    mean_time = sum(per_prompt_times) / len(per_prompt_times)
    std_time = (
        (sum((x - mean_time) ** 2 for x in per_prompt_times) / len(per_prompt_times)) ** 0.5
        if len(per_prompt_times) > 1 else 0.0
    )

    return {
        "n_prompts": len(prompts),
        "total_tokens": total_tokens,
        "total_time_ms": round(total_time_ms, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "ms_per_token": round(ms_per_token, 4),
        "mean_ms_per_prompt": round(mean_time, 2),
        "std_ms_per_prompt": round(std_time, 2),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    smoke = "--smoke" in sys.argv
    n_prompts = 10 if smoke else 100
    n_runs = 1 if smoke else 3

    # Parse --n and --runs flags
    for i, arg in enumerate(sys.argv):
        if arg == "--n" and i + 1 < len(sys.argv):
            n_prompts = int(sys.argv[i + 1])
        if arg == "--runs" and i + 1 < len(sys.argv):
            n_runs = int(sys.argv[i + 1])

    print("=" * 70)
    print("XP-36c: Latency Overhead — Baseline vs TEL-OS SLERP")
    print("=" * 70)
    print(f"  Modelo:        meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(f"  max_new_tokens: {XP36cConfig().max_new_tokens} (fijo)")
    print(f"  Prompts:       {n_prompts} por condición")
    print(f"  Runs:          {n_runs} por condición (reportar mean ± std)")
    print(f"  Timing:        {'CUDA events (GPU precision)' if torch.cuda.is_available() else 'wall clock'}")
    print(f"  Smoke:         {smoke}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = XP36cConfig()
    prompts = LATENCY_PROMPTS[:n_prompts]

    # Model
    print(f"\n[1/3] Cargando {config.model_name} en {device}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  [OK] {len(model.model.layers)} capas en {device}")

    # Vectors
    print(f"\n[2/3] Cargando vectores TEL-OS...")
    if not VECTORS_PATH.exists():
        raise RuntimeError(f"FATAL: {VECTORS_PATH} no encontrado.")
    all_vectors = torch.load(VECTORS_PATH, map_location=device)
    model_dtype = next(model.parameters()).dtype

    v_early_n = F.normalize(all_vectors[config.detection_layer_early].to(model_dtype), dim=0)
    v_late_n = F.normalize(all_vectors[config.detection_layer_late].to(model_dtype), dim=0)
    steering_vecs = {
        l: F.normalize(all_vectors[l].to(model_dtype), dim=0)
        for l in config.steering_layers if l in all_vectors
    }
    print(f"  [OK] Steering layers: {config.steering_layers}")

    # TEL-OS hooks factory
    urgency_active = torch.zeros(1, device=device)

    def telos_hooks_fn():
        urgency_active[0] = 0.0
        hooks = []

        def det_hook(vec_n):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                if h.shape[1] == 1:  # generación token-a-token → skip detection
                    return output
                raw_d = torch.dot(
                    F.normalize(h[:, -1, :][0].float(), dim=0), vec_n.float()
                ).item()
                u = 1.0 + max(raw_d - config.urgency_threshold, 0) * 200.0
                if u > float(urgency_active[0]):
                    urgency_active[0] = min(u, config.urgency_cap)
                return output
            return hook

        def steer_hook(vec_n):
            def hook(module, input, output):
                alpha = min(float(urgency_active[0]) * config.alpha_max, 1.0)
                if alpha < 1e-6:
                    return output
                h = output[0] if isinstance(output, tuple) else output
                h_s = slerp_steer(h, vec_n.to(h.dtype), alpha)
                return (h_s,) + output[1:] if isinstance(output, tuple) else h_s
            return hook

        def decay_hook():
            def hook(module, input, output):
                if float(urgency_active[0]) > 1.0:
                    h = output[0] if isinstance(output, tuple) else output
                    h = h * config.decay_factor
                    return (h,) + output[1:] if isinstance(output, tuple) else h
                return output
            return hook

        hooks.append(model.model.layers[config.detection_layer_early].register_forward_hook(
            det_hook(v_early_n)
        ))
        hooks.append(model.model.layers[config.detection_layer_late].register_forward_hook(
            det_hook(v_late_n)
        ))
        for l, v in steering_vecs.items():
            hooks.append(model.model.layers[l].register_forward_hook(steer_hook(v)))
        decay_layer = min(config.detection_layer_late + 1, len(model.model.layers) - 1)
        hooks.append(model.model.layers[decay_layer].register_forward_hook(decay_hook()))
        return hooks

    # Warmup (descartado)
    print(f"\n[3/3] Midiendo latencia...")
    print(f"  Warmup ({config.warmup_prompts} prompts)...")
    warmup_prompts = prompts[:config.warmup_prompts]
    measure_generation(model, tokenizer, warmup_prompts, config, device)
    print("  [OK] Warmup completado")

    # Run measurements
    baseline_runs = []
    telos_runs = []

    for run in range(n_runs):
        print(f"\n  Run {run + 1}/{n_runs}:")

        # Baseline
        print(f"    Baseline (sin hooks)...", end=" ", flush=True)
        b = measure_generation(model, tokenizer, prompts, config, device)
        baseline_runs.append(b)
        print(f"{b['tokens_per_sec']:.1f} tok/s ({b['ms_per_token']:.3f} ms/tok)")

        # TEL-OS
        print(f"    TEL-OS SLERP (con hooks)...", end=" ", flush=True)
        t = measure_generation(model, tokenizer, prompts, config, device, hooks_fn=telos_hooks_fn)
        telos_runs.append(t)
        print(f"{t['tokens_per_sec']:.1f} tok/s ({t['ms_per_token']:.3f} ms/tok)")

    # Aggregate runs
    def mean_std(values):
        m = sum(values) / len(values)
        s = (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5 if len(values) > 1 else 0.0
        return m, s

    b_tps_vals = [r["tokens_per_sec"] for r in baseline_runs]
    t_tps_vals = [r["tokens_per_sec"] for r in telos_runs]
    b_mpt_vals = [r["ms_per_token"] for r in baseline_runs]
    t_mpt_vals = [r["ms_per_token"] for r in telos_runs]
    b_total_vals = [r["total_time_ms"] / 1000.0 for r in baseline_runs]  # convert to seconds
    t_total_vals = [r["total_time_ms"] / 1000.0 for r in telos_runs]

    b_tps_mean, b_tps_std = mean_std(b_tps_vals)
    t_tps_mean, t_tps_std = mean_std(t_tps_vals)
    b_mpt_mean, b_mpt_std = mean_std(b_mpt_vals)
    t_mpt_mean, t_mpt_std = mean_std(t_mpt_vals)
    b_total_mean, b_total_std = mean_std(b_total_vals)
    t_total_mean, t_total_std = mean_std(t_total_vals)

    overhead_tps = (b_tps_mean - t_tps_mean) / b_tps_mean * 100 if b_tps_mean > 0 else 0.0
    overhead_total = (t_total_mean - b_total_mean) / b_total_mean * 100 if b_total_mean > 0 else 0.0
    paper_ready = overhead_tps < 20.0

    summary = {
        "experiment": "XP-36c",
        "date": datetime.now().isoformat(),
        "model": config.model_name,
        "telos_version": "v3.0-SLERP",
        "alpha_max": config.alpha_max,
        "max_new_tokens": config.max_new_tokens,
        "n_prompts": n_prompts,
        "n_runs": n_runs,
        "timing_method": "CUDA events" if torch.cuda.is_available() else "wall clock",
        "device": device,
        "baseline": {
            "tokens_per_sec_mean": round(b_tps_mean, 2),
            "tokens_per_sec_std": round(b_tps_std, 2),
            "ms_per_token_mean": round(b_mpt_mean, 4),
            "ms_per_token_std": round(b_mpt_std, 4),
            "total_time_s_mean": round(b_total_mean, 2),
            "total_time_s_std": round(b_total_std, 2),
            "raw_runs": baseline_runs,
        },
        "telos": {
            "tokens_per_sec_mean": round(t_tps_mean, 2),
            "tokens_per_sec_std": round(t_tps_std, 2),
            "ms_per_token_mean": round(t_mpt_mean, 4),
            "ms_per_token_std": round(t_mpt_std, 4),
            "total_time_s_mean": round(t_total_mean, 2),
            "total_time_s_std": round(t_total_std, 2),
            "raw_runs": telos_runs,
        },
        "overhead": {
            "tokens_per_sec_loss_pct": round(overhead_tps, 2),
            "total_time_overhead_pct": round(overhead_total, 2),
        },
        "paper_ready": paper_ready,
        "success_criterion": "overhead < 20% in tokens/sec",
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("XP-36c — RESULTADOS FINALES")
    print("=" * 70)
    print(f"  {'Métrica':<35} {'Baseline':>14} {'TEL-OS':>14} {'Overhead':>10}")
    print(f"  {'─'*35} {'─'*14} {'─'*14} {'─'*10}")
    print(f"  {'tokens/sec (mean ± std)':<35} {b_tps_mean:>8.1f}±{b_tps_std:>4.1f} {t_tps_mean:>8.1f}±{t_tps_std:>4.1f} {overhead_tps:>9.2f}%")
    print(f"  {'ms/token (mean ± std)':<35} {b_mpt_mean:>9.3f}±{b_mpt_std:>3.3f} {t_mpt_mean:>9.3f}±{t_mpt_std:>3.3f} {'—':>10}")
    print(f"  {f'Total time {n_prompts} prompts (s)':<35} {b_total_mean:>9.1f}±{b_total_std:>3.1f} {t_total_mean:>9.1f}±{t_total_std:>3.1f} {overhead_total:>9.2f}%")
    print(f"\n  Overhead tokens/sec: {overhead_tps:.2f}% {'✅' if paper_ready else '❌'} (criterio: <20%)")
    print(f"  Paper-ready: {'✅ SÍ' if paper_ready else '❌ NO — revisar alpha_max o hooks'}")
    print(f"\n  Resultados: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
