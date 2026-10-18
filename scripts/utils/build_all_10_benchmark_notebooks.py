#!/usr/bin/env python3
"""
Master Builder for NeuralOps Benchmark Notebooks (10-19).
Author: Taha Majlesi
"""

import sys
from pathlib import Path

# Ensure local script imports work
scripts_utils_dir = Path(__file__).resolve().parent
if str(scripts_utils_dir) not in sys.path:
    sys.path.insert(0, str(scripts_utils_dir))

from nb_gen_common import save_notebook
from nb_gen_10_12 import generate_notebook_10, generate_notebook_11, generate_notebook_12
from nb_gen_13_15 import generate_notebook_13, generate_notebook_14, generate_notebook_15
from nb_gen_16_17 import generate_notebook_16, generate_notebook_17
from nb_gen_18_19 import generate_notebook_18, generate_notebook_19

NOTEBOOK_BUILDERS = [
    ("notebooks/10_long_context_extrapolation_32k.ipynb", generate_notebook_10),
    ("notebooks/11_cuda_vs_triton_vs_pytorch_matrix.ipynb", generate_notebook_11),
    ("notebooks/12_svd_whitening_spectral_allocation.ipynb", generate_notebook_12),
    ("notebooks/13_quantization_pareto_fp8_int4_kmeans.ipynb", generate_notebook_13),
    ("notebooks/14_structured_pruning_token_shedding.ipynb", generate_notebook_14),
    ("notebooks/15_downstream_glue_lra_evaluation.ipynb", generate_notebook_15),
    ("notebooks/16_roofline_ada_rtx4070_saturation.ipynb", generate_notebook_16),
    ("notebooks/17_speculative_paged_kv_throughput.ipynb", generate_notebook_17),
    ("notebooks/18_cross_seed_statistical_variance.ipynb", generate_notebook_18),
    ("notebooks/19_publication_figures_latex_generator.ipynb", generate_notebook_19),
]


def main():
    """Execute main routine."""
    print("=" * 80)
    print("🚀 NeuralOps Benchmark Notebooks (10-19) Generation Master")
    print("=" * 80)

    repo_root = Path(__file__).resolve().parent.parent.parent

    for rel_path, builder_fn in NOTEBOOK_BUILDERS:
        target_path = repo_root / rel_path
        print(f"\nBuilding: {rel_path} ...")
        nb = builder_fn()
        save_notebook(nb, target_path)

    print("\n" + "=" * 80)
    print("✅ All 10 Benchmark Notebooks successfully generated!")
    print("=" * 80)


if __name__ == "__main__":
    main()
