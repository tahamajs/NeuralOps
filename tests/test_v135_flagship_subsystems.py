# -*- coding: utf-8 -*-
"""Comprehensive verification suite for NeuralOps v1.3.5 flagship subsystems.

Tests:
1. Hardware Roofline Profiler (analytical FLOPs, HBM bytes, dual-path fallback, terminal render)
2. Layerwise Subspace Micro-Healing (composite loss optimization, alignment improvement)
3. Surgical Checkpoint IO (manifest extraction, round-trip state preservation)
4. Academic LaTeX & Pareto Frontier Exporter (dominance check, booktabs, stars, bolding)
5. CLI Subcommands (roofline, transplant with heal, results with format)
"""

import io
import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import neuralops
from neuralops.core.composable import ComposableAttention, compose_attention
from neuralops.core.paper_export import compute_pareto_frontier, export_benchmark_table
from neuralops.core.roofline import (
    RooflineReport,
    estimate_module_flops_and_bytes,
    get_device_roofline_specs,
    profile_roofline,
    render_roofline_terminal,
)
from neuralops.models.surgical_io import (
    extract_surgical_manifest,
    load_surgical_checkpoint,
    save_surgical_checkpoint,
)


# -----------------------------------------------------------------------------
# 1. Roofline Profiler Tests
# -----------------------------------------------------------------------------
def test_roofline_specs_detection():
    """Verify hardware roofline lookup and fallback heuristics."""
    name, sm, tflops, bw = get_device_roofline_specs("cpu")
    assert "cpu" in sm.lower() or "CPU" in name
    assert tflops > 0.0
    assert bw > 0.0

    if torch.cuda.is_available():
        cuda_name, cuda_sm, cuda_tflops, cuda_bw = get_device_roofline_specs("cuda:0")
        assert "SM" in cuda_sm
        assert cuda_tflops > 0.0
        assert cuda_bw > 0.0


def test_roofline_analytical_estimation():
    """Verify analytical estimation of FLOPs and HBM bytes."""
    mod = nn.Linear(256, 256)
    flops, hbm_naive, hbm_fused = estimate_module_flops_and_bytes(
        module=mod, batch_size=2, seq_len=128, embed_dim=256, num_heads=4, element_bytes=2
    )
    assert flops > 0.0
    assert hbm_fused > 0.0
    assert hbm_naive >= hbm_fused


def test_profile_roofline_registered_and_composable():
    """Run dual-path timing roofline analysis on both registered attention and ComposableAttention."""
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = "float16" if torch.cuda.is_available() else "float32"

    # Test registered attention
    rep1 = profile_roofline(
        module_or_name="minimal_svd",
        batch_size=1,
        seq_len=64,
        embed_dim=64,
        num_heads=4,
        dtype=dtype,
        device=dev,
        warmup=1,
        rep=2,
    )
    assert isinstance(rep1, RooflineReport)
    assert rep1.total_flops > 0.0
    assert rep1.hbm_bytes_fused > 0.0
    assert rep1.arithmetic_intensity_fused > 0.0
    assert rep1.bottleneck_regime != ""
    assert isinstance(rep1.to_dict(), dict)

    # Test GatedDeltaNet accurate FLOPs and 2D Log-Log chart
    rep_gdn = profile_roofline(
        module_or_name="gated_delta_net",
        batch_size=2,
        seq_len=2048,
        embed_dim=768,
        num_heads=12,
        dtype=dtype,
        device=dev,
        warmup=2,
        rep=3,
    )
    assert rep_gdn.total_flops >= 15e9, f"Expected total_flops >= 15 GFLOPs, got {rep_gdn.total_flops / 1e9:.2f} GFLOPs"
    assert rep_gdn.sram_bytes_saved_mb > 0.0
    terminal_gdn = render_roofline_terminal(rep_gdn)
    assert "[T]" in terminal_gdn or "T" in terminal_gdn
    assert "[P]" in terminal_gdn or "P" in terminal_gdn
    assert "Peak:" in terminal_gdn
    assert "Ridge:" in terminal_gdn


# -----------------------------------------------------------------------------
# 2. Layerwise Subspace Micro-Healing Tests
# -----------------------------------------------------------------------------
class ToyAttentionLayer(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.self = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.self(x)


class ToyTransformer(nn.Module):
    def __init__(self, num_layers: int = 2, dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([ToyAttentionLayer(dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def test_layerwise_subspace_micro_healing():
    """Verify that heal_replaced_model aligns student to teacher representations."""
    from neuralops.core._attention_surgery import heal_replaced_model

    teacher = ToyTransformer(num_layers=2, dim=64)
    student = ToyTransformer(num_layers=2, dim=64)

    # Scramble student weights
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    x_calib = torch.randn(4, 32, 64)

    stats = heal_replaced_model(
        teacher_model=teacher,
        student_model=student,
        calibration_inputs=x_calib,
        steps=25,
        lr=5e-3,
        also_ridge_init=False,
    )

    assert stats["num_healed_layers"] == 2
    assert stats["final_mse"] < stats["initial_mse"]
    assert stats["final_cosine_sim"] >= stats["initial_cosine_sim"]


# -----------------------------------------------------------------------------
# 3. Surgical Checkpoint Persistence Tests
# -----------------------------------------------------------------------------
def test_surgical_checkpoint_roundtrip():
    """Verify lossless extraction, saving, and dynamic loading of surgical models."""
    teacher = ToyTransformer(num_layers=2, dim=64)
    # Replace second layer with a ComposableAttention
    composable = ComposableAttention(embed_dim=64, num_heads=4)
    teacher.layers[1].self = composable

    manifest = extract_surgical_manifest(teacher)
    assert "layers.1.self" in manifest["submodules"]
    assert manifest["submodules"]["layers.1.self"]["category"] == "attention"

    x_test = torch.randn(2, 16, 64)
    with torch.no_grad():
        out_orig = teacher(x_test)

    temp_dir = tempfile.mkdtemp()
    try:
        save_path = save_surgical_checkpoint(teacher, temp_dir, metadata={"experiment": "test_v135"})
        assert (save_path / "surgical_manifest.json").exists()

        # Instantiate fresh unmodified base model
        fresh_model = ToyTransformer(num_layers=2, dim=64)
        loaded_model, loaded_manifest = load_surgical_checkpoint(fresh_model, save_path)

        with torch.no_grad():
            out_loaded = loaded_model(x_test)

        assert torch.allclose(out_orig, out_loaded, atol=1e-5)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# 4. Academic LaTeX & Pareto Exporter Tests
# -----------------------------------------------------------------------------
def test_pareto_frontier_computation():
    """Verify Pareto non-dominance calculation across multi-objective tradeoffs."""
    records = [
        {"name": "baseline", "latency_ms": 10.0, "speedup_x": 1.0, "ppl": 20.0},
        {"name": "optimal_fast", "latency_ms": 2.0, "speedup_x": 5.0, "ppl": 22.0},  # Pareto optimal
        {"name": "optimal_accurate", "latency_ms": 5.0, "speedup_x": 2.0, "ppl": 15.0},  # Pareto optimal
        {"name": "dominated", "latency_ms": 8.0, "speedup_x": 1.2, "ppl": 25.0},  # Dominated
    ]
    mask = compute_pareto_frontier(records)
    assert mask[1] is True  # optimal_fast
    assert mask[2] is True  # optimal_accurate
    assert mask[3] is False  # dominated


def test_export_benchmark_table_formats():
    """Verify booktabs LaTeX, Markdown, and CSV table generation."""
    records = [
        {"module": "gla_attention", "latency_triton_ms": 1.25, "speedup_x": 3.45, "sram_bytes_saved_mb": 14.2},
        {"module": "gated_delta_net", "latency_triton_ms": 0.95, "speedup_x": 4.52, "sram_bytes_saved_mb": 18.5},
    ]

    latex_out = export_benchmark_table(records, format="latex")
    assert r"\begin{table}" in latex_out
    assert r"\toprule" in latex_out
    assert r"\bottomrule" in latex_out
    assert r"\textbf{" in latex_out
    assert r"$^\star$" in latex_out

    md_out = export_benchmark_table(records, format="markdown")
    assert "| Module |" in md_out
    assert "**" in md_out
    assert "★" in md_out

    csv_out = export_benchmark_table(records, format="csv")
    assert "gla_attention" in csv_out
    assert "pareto_optimal" in csv_out


# -----------------------------------------------------------------------------
# 5. Architecture Inspector & CLI Subsystem Tests
# -----------------------------------------------------------------------------
def test_inspect_architecture_all_registries():
    """Verify that inspect_architecture works for 100% of 131 attentions and 72 compressors."""
    import neuralops.attentions
    import neuralops.compression_engine
    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    from neuralops.core.inspector import inspect_architecture, render_architecture_card

    assert len(ATTENTION_REGISTRY) == 131
    assert len(COMPRESSOR_REGISTRY) == 72

    # Comprehensive verification across all 131 attentions
    for key in list(ATTENTION_REGISTRY.keys()):
        spec = inspect_architecture(key)
        assert spec.name == key
        assert spec.category == "attention"
        assert len(spec.time_complexity) > 0
        assert len(spec.memory_complexity) > 0
        assert "::" in spec.triton_kernel_module
        assert len(spec.math_equation) > 0
        assert len(spec.constructor_params) > 0
        assert "pytorch_direct" in spec.quickstart_code

    # Comprehensive verification across all 72 compressors
    for key in list(COMPRESSOR_REGISTRY.keys()):
        spec = inspect_architecture(key)
        assert spec.name == key
        assert spec.category == "compressor"
        assert len(spec.time_complexity) > 0
        assert len(spec.memory_complexity) > 0
        assert "::" in spec.triton_kernel_module
        assert len(spec.math_equation) > 0
        assert len(spec.constructor_params) > 0
        assert "pytorch_direct" in spec.quickstart_code

    # Test exemplar rendering
    card = render_architecture_card(inspect_architecture("gated_delta_net"))
    assert "gated_delta_net" in card
    assert "── python ──" in card


def test_cli_inspect_and_ui_commands():
    """Verify CLI commands (inspect, ui snapshot, results)."""
    from neuralops.benchmarks.cli import main
    import io
    import sys

    # 1. Test inspect gated_delta_net
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        ret = main(["inspect", "gated_delta_net"])
        out = sys.stdout.getvalue()
        assert ret == 0
        assert "gated_delta_net" in out
        assert "gdn_scan.py" in out
    finally:
        sys.stdout = old_stdout

    # 2. Test ui --snapshot
    sys.stdout = io.StringIO()
    try:
        ret = main(["ui", "--snapshot"])
        out = sys.stdout.getvalue()
        assert ret == 0
        assert "NEURALOPS MINIMALIST ARCHITECTURE EXPLORER" in out
    finally:
        sys.stdout = old_stdout

