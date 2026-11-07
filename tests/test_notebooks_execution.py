#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Two-level execution and structural validation test suite for NeuralOps Jupyter notebooks (F-75).

Level (a) SMOKE & STRUCTURE (CI, CPU-friendly):
- Validates JSON schema and AST syntactic correctness for all 94 unique notebooks.
- Executes fast CPU smoke tests on notebooks 01, 02, and 03 with clean registry isolation.

Level (b) FULL HARDWARE EXECUTION (GPU / Sharif HPC):
- Full end-to-end execution of the 29 validated notebooks via nbconvert or isolated kernel runner.
- Marked with @pytest.mark.gpu and skipped when CUDA is unavailable.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import warnings
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Master list of all 94 canonical notebooks
ALL_NOTEBOOKS = sorted([p.name for p in NOTEBOOKS_DIR.glob("*.ipynb")])

# 29 Authoritatively Executed Notebooks on NVIDIA RTX 3090 (Sharif HPC)
EXECUTED_29_NOTEBOOKS = [
    "00_master_neuralops_complete_system_walkthrough.ipynb",
    "01_attention_surgery_and_lora.ipynb",
    "01_quickstart_surgery_and_healing.ipynb",
    "02_roofline_and_triton_kernels.ipynb",
    "02_unified_training.ipynb",
    "03_custom_plugins_and_extensibility.ipynb",
    "04_end_to_end_3level_benchmarking.ipynb",
    "05_master_108_cuda_and_26_triton_kernels_benchmark.ipynb",
    "05_paper_empirical_results_master.ipynb",
    "06_master_all_131_attention_mechanisms_interactive.ipynb",
    "07_master_all_72_compression_engines_quant_and_svd.ipynb",
    "08_six_pillar_distillation_and_surgical_recovery.ipynb",
    "09_production_serving_paged_cache_and_speculative_decoding.ipynb",
    "10_long_context_extrapolation_32k.ipynb",
    "11_cuda_vs_triton_vs_pytorch_matrix.ipynb",
    "12_svd_whitening_spectral_allocation.ipynb",
    "13_quantization_pareto_fp8_int4_kmeans.ipynb",
    "14_structured_pruning_token_shedding.ipynb",
    "15_downstream_glue_lra_evaluation.ipynb",
    "16_roofline_ada_rtx4070_saturation.ipynb",
    "17_speculative_paged_kv_throughput.ipynb",
    "18_cross_seed_statistical_variance.ipynb",
    "19_publication_figures_latex_generator.ipynb",
    "20_production_serving_vllm_sglang_trtllm_integrations.ipynb",
    "21_model_export_onnx_safetensors_surgery_manifest.ipynb",
    "22_distributed_tensor_and_context_parallelism.ipynb",
    "23_spectral_diagnostics_and_hessian_analysis.ipynb",
    "24_advanced_optimizers_muon_stiefel_and_ttt_training.ipynb",
    "25_long_context_niah_cka_similarity_and_eval_harness.ipynb",
]


def _execute_notebook_cells(nb_path: Path):
    """Parse and execute code cells from an .ipynb file in sequence for CPU smoke tests."""
    assert nb_path.exists(), f"Notebook not found: {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    # Ensure headless matplotlib backend to avoid GUI warnings in CI
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.show = lambda *args, **kwargs: None
    except ImportError:
        pass

    exec_globals = {
        "__name__": "__main__",
        "__file__": str(nb_path),
        "display": lambda *args, **kwargs: None,
    }

    for idx, cell in enumerate(nb_data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        lines = cell.get("source", [])
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("%", "!")):
                continue
            code_lines.append(line)

        code_str = "".join(code_lines)
        if not code_str.strip():
            continue

        try:
            exec(code_str, exec_globals)
        except Exception as e:
            pytest.fail(f"Execution failed in {nb_path.name} cell #{idx}:\nCode:\n{code_str}\nError: {e}")


# =========================================================================
# Tier 1 (a): Structure and AST Syntax Validation across ALL 94 Notebooks
# =========================================================================

@pytest.mark.parametrize("nb_name", ALL_NOTEBOOKS)
def test_notebook_structure_and_syntax(nb_name: str):
    """Verify that every notebook has valid JSON schema and all code cells parse with Python AST."""
    nb_path = NOTEBOOKS_DIR / nb_name
    assert nb_path.exists(), f"Notebook file {nb_name} does not exist"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data, f"{nb_name} missing 'cells' array"
    assert "metadata" in nb_data, f"{nb_name} missing 'metadata'"
    assert "nbformat" in nb_data, f"{nb_name} missing 'nbformat'"

    for idx, cell in enumerate(nb_data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        lines = [line for line in cell.get("source", []) if not line.strip().startswith(("%", "!"))]
        code = "".join(lines)
        if not code.strip():
            continue

        with warnings.catch_warnings():
            # In Python 3.12+, invalid escapes in LaTeX strings emit SyntaxWarning; ignore during AST parse
            warnings.simplefilter("ignore", SyntaxWarning)
            try:
                ast.parse(code)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {nb_name} cell #{idx} (line {e.lineno}):\n{e.text}\n{e}")


# =========================================================================
# Tier 1 (b): Fast CPU Smoke Execution of Core Representative Notebooks
# =========================================================================

def test_notebook_smoke_01_quickstart_surgery_and_healing():
    """Verify end-to-end CPU smoke execution of Notebook 01."""
    nb_path = NOTEBOOKS_DIR / "01_quickstart_surgery_and_healing.ipynb"
    _execute_notebook_cells(nb_path)


def test_notebook_smoke_02_roofline_and_triton_kernels():
    """Verify end-to-end CPU smoke execution of Notebook 02."""
    nb_path = NOTEBOOKS_DIR / "02_roofline_and_triton_kernels.ipynb"
    _execute_notebook_cells(nb_path)


def test_notebook_smoke_03_custom_plugins_and_extensibility():
    """Verify end-to-end CPU smoke execution of Notebook 03 and restore registry purity."""
    from neuralops.core.extensibility import unregister_custom_module
    from neuralops.core.registry import ATTENTION_REGISTRY

    nb_path = NOTEBOOKS_DIR / "03_custom_plugins_and_extensibility.ipynb"
    initial_count = len(ATTENTION_REGISTRY)
    try:
        _execute_notebook_cells(nb_path)
    finally:
        if "my_research_attn" in ATTENTION_REGISTRY:
            unregister_custom_module("my_research_attn")
        if "custom_spectral_probe_attn" in ATTENTION_REGISTRY:
            unregister_custom_module("custom_spectral_probe_attn")
        assert len(ATTENTION_REGISTRY) == initial_count, f"Expected {initial_count} attentions, got {len(ATTENTION_REGISTRY)}"


# =========================================================================
# Tier 3 / 5: Full GPU Execution of 29 Evaluated Notebooks (Sharif HPC)
# =========================================================================

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full execution requires CUDA GPU on HPC cluster")
@pytest.mark.parametrize("nb_name", EXECUTED_29_NOTEBOOKS)
def test_notebook_full_execution_gpu(nb_name: str):
    """Full hardware execution on NVIDIA GPU (RTX 3090) with per-notebook execution gate."""
    nb_path = NOTEBOOKS_DIR / nb_name
    _execute_notebook_cells(nb_path)
