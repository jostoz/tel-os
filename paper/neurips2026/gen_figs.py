"""
Generate all paper figures WITHOUT embedded titles.
Output: paper/neurips2026/assets/

Run from BISON root:
    C:/Users/joz/Desktop/OBLITERATUS/venv/Scripts/python.exe paper/neurips2026/gen_figs.py
"""
import os
import sys
from pathlib import Path

# ── Change to BISON root so relative data paths work ────────────────────────
ROOT = Path(__file__).parent.parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = ROOT / "paper" / "neurips2026" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Monkey-patch: strip all embedded titles ──────────────────────────────────
plt.Axes.set_title = lambda self, *a, **kw: None
plt.Figure.suptitle = lambda self, *a, **kw: None

# ── Monkey-patch: redirect savefig to OUT_DIR ────────────────────────────────
_orig_savefig = plt.savefig

def _new_savefig(fname, *args, **kwargs):
    dest = OUT_DIR / Path(str(fname)).name
    print(f"  -> {dest.name}")
    _orig_savefig(str(dest), *args, **kwargs)

plt.savefig = _new_savefig

# ── Helper: exec a script with correct __file__ ─────────────────────────────
def run_script(rel_path):
    abs_path = str(ROOT / rel_path)
    gl = {
        "__file__": abs_path,
        "__name__": "__script__",
    }
    print(f"\n[gen] {rel_path}")
    exec(open(abs_path, encoding="utf-8").read(), gl)  # noqa: S102


# ── Fig 1, 2, 3 — from scratch/generate_paper_figures.py ────────────────────
print("\n=== Fig 1 / 2 / 3 ===")
gl_gen = {
    "__file__": str(ROOT / "scratch" / "generate_paper_figures.py"),
    "__name__": "__script__",
}
exec(open(ROOT / "scratch" / "generate_paper_figures.py", encoding="utf-8").read(), gl_gen)

gl_gen["make_fig1"]()
print("  fig1 done")
gl_gen["make_fig2"]()
print("  fig2 done")
gl_gen["make_fig3"]()
print("  fig3 done")

# ── Fig 4–9 — individual scripts ────────────────────────────────────────────
print("\n=== Fig 4 – 9 ===")
for script in [
    "experiments/local/fig4_spider_crossmodel.py",
    "experiments/local/fig5_slerp_geometry.py",
    "experiments/local/fig6_alpha_ablation.py",
    "experiments/local/fig7_harmbench_category.py",
    "experiments/local/fig8_autodan_flatline.py",
    "experiments/local/fig9_asr_heatmap.py",
]:
    run_script(script)

print(f"\n[done] All figures saved to {OUT_DIR}")
