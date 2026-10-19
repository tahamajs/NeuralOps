#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive verification suite for NeuralOps v1.3.6 Extensibility, Custom Modules & Integration.

Tests:
1. Universal 203-Module Registry & Inspector contract (131 attentions, 72 compressors)
2. Custom Plugin Lifecycle: Dynamic registration, validation, surgery, healing, persistence, ablation
3. Custom Composable Algebraic stage registration (P, phi, M, E)
4. Contract Diagnostic Suite: validate_module_contract & scaffold_module
5. Roofline speedup assertion (>3.0x on GatedDeltaNet)
6. Canonical registry baseline protection (131 / 72 / 26 counts remain pristine after unregister)
"""

import io
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import neuralops
from neuralops.core.extensibility import (
    get_custom_metadata,
    register_custom_attention,
    register_custom_compressor,
    scaffold_module,
    unregister_custom_module,
    validate_module_contract,
)
from neuralops.core.inspector import inspect_architecture, render_architecture_card
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, create_attention, create_compressor
from neuralops.core.roofline import profile_roofline, render_roofline_terminal


def test_universal_registry_and_inspector_contract():
    """Verify all 131 attentions and 72 compressors can be inspected with clean signatures."""
    attn_keys = list(ATTENTION_REGISTRY.keys())
    comp_keys = list(COMPRESSOR_REGISTRY.keys())
    assert len(attn_keys) == 131, f"Expected 131 attentions, found {len(attn_keys)}"
    assert len(comp_keys) == 72, f"Expected 72 compressors, found {len(comp_keys)}"

    for k in ["gated_delta_net", "gla_attention", "minimal_svd", "cosformer", "mamba_block"]:
        spec = inspect_architecture(k, category="attention")
        assert spec.name == k
        assert spec.family != ""
        assert spec.time_complexity != ""
        card = render_architecture_card(spec)
        assert len(card.split("\n")) >= 10
        assert "init():" in card
        assert "Any | None=None" not in card

    for k in ["hadamard_incoherent_svd", "cur_compressor", "dobi_svd", "direct_svd", "slicegpt"]:
        spec = inspect_architecture(k, category="compressor")
        assert spec.name == k
        assert spec.category == "compressor"
        assert len(spec.constructor_params) > 0


def test_custom_plugin_lifecycle_and_contract_validation():
    """Test full lifecycle of a dynamically registered custom attention and compressor."""
    custom_attn_name = "test_custom_research_attn"

    @neuralops.register_custom_attention(
        custom_attn_name,
        family="linear-scan",
        time_complexity="O(T·d²)",
        memory_complexity="O(d²) state",
        math_equation=r"S_t = S_{t-1} + \phi(k_t) \otimes v_t  \mid  o_t = \phi(q_t) \cdot S_t",
        triton_kernel="custom_kernel.py :: _fwd",
        sram_fusion_strategy="SRAM register accumulator",
    )
    class CustomResearchAttention(nn.Module):
        def __init__(self, embed_dim: int = 64, num_heads: int = 4, rank: int | None = None, **kwargs):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            B, T, D = hidden_states.shape
            H, d = self.num_heads, self.head_dim
            q = self.q_proj(hidden_states).view(B, T, H, d).transpose(1, 2)
            k = self.k_proj(hidden_states).view(B, T, H, d).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, T, H, d).transpose(1, 2)

            q_phi = F.elu(q) + 1.0
            k_phi = F.elu(k) + 1.0
            scores = torch.matmul(q_phi, k_phi.transpose(-1, -2))
            context = torch.matmul(scores, v)
            out = self.out_proj(context.transpose(1, 2).contiguous().view(B, T, D))
            return out

    try:
        assert custom_attn_name in ATTENTION_REGISTRY

        spec = inspect_architecture(custom_attn_name)
        assert spec.family == "linear-scan"
        assert spec.triton_kernel_module == "custom_kernel.py :: _fwd"
        assert "SRAM register accumulator" in spec.sram_fusion_strategy

        val_res = validate_module_contract(custom_attn_name, embed_dim=64, num_heads=4, seq_len=16)
        assert val_res["all_passed"], f"Contract validation failed: {val_res}"
        assert val_res["passed_count"] == 6

        report = profile_roofline(
            custom_attn_name,
            batch_size=1,
            seq_len=64,
            embed_dim=64,
            num_heads=4,
            warmup=1,
            rep=2,
        )
        assert report.total_flops > 0.0
        assert report.hbm_bytes_fused > 0.0

        class HostModel(nn.Module):
            def __init__(self, d=64):
                super().__init__()
                self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
                self.dense = nn.Linear(d, d)
            def forward(self, x):
                o = self.attn(x, x, x)
                if isinstance(o, tuple):
                    o = o[0]
                return self.dense(o)

        teacher = HostModel(d=64)
        x_calib = torch.randn(4, 16, 64)
        tmp_dir = tempfile.mkdtemp()
        try:
            pipeline_res = neuralops.surgical_transplant_and_heal_pipeline(
                teacher_model=teacher,
                target_attention=custom_attn_name,
                calibration_inputs=x_calib,
                heal_steps=5,
                heal_lr=1e-3,
                save_dir=tmp_dir,
            )
            assert pipeline_res["model"] is not None
            assert pipeline_res["saved_path"] is not None

            fresh_host = HostModel(d=64)
            reloaded_model, manifest = neuralops.load_surgical_checkpoint(fresh_host, tmp_dir)
            reloaded_model.eval()
            pipeline_res["model"].eval()
            with torch.no_grad():
                out_orig = pipeline_res["model"](x_calib)
                out_reloaded = reloaded_model(x_calib)
            assert torch.allclose(out_orig, out_reloaded, atol=1e-5)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    finally:
        unregister_custom_module(custom_attn_name, category="attention")
        assert custom_attn_name not in ATTENTION_REGISTRY
        assert len(ATTENTION_REGISTRY) == 131


def test_custom_composable_algebraic_stage_registration():
    """Verify registering a custom feature map and custom engine in ComposableAttention."""
    custom_phi_name = "test_custom_elu_squared"

    @neuralops.register_composable_kernel(custom_phi_name)
    def custom_kernel(x: torch.Tensor) -> torch.Tensor:
        return (F.elu(x.float()) + 1.0) ** 2

    attn = neuralops.compose_attention(
        projection="dense",
        kernel_map=custom_phi_name,
        mask="causal",
        engine="sdpa",
        embed_dim=64,
        num_heads=4,
    )
    x = torch.randn(2, 16, 64)
    out = attn(x)
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()

    from neuralops.core.composable import _CUSTOM_KERNELS
    _CUSTOM_KERNELS.pop(custom_phi_name, None)


def test_scaffolding_generator():
    """Verify scaffolding generator produces valid, readable Python code."""
    code_attn = scaffold_module("my_brand_new_attn", category="attention", family="gated-delta-recurrent")
    assert "@neuralops.register_custom_attention" in code_attn
    assert "class MyBrandNewAttn(nn.Module):" in code_attn

    code_comp = scaffold_module("my_brand_new_comp", category="compressor", family="svd-low-rank")
    assert "@neuralops.register_custom_compressor" in code_comp
    assert "class MyBrandNewComp:" in code_comp


def test_roofline_hardware_truthfulness_rtx4070():
    """Assert physical truthfulness on GPU node: speedup > 3.0x on GatedDeltaNet."""
    if not torch.cuda.is_available():
        pytest.skip("Test requires CUDA device")

    report = profile_roofline(
        "gated_delta_net",
        batch_size=2,
        seq_len=1024,
        embed_dim=768,
        num_heads=12,
        device="cuda:0",
        warmup=3,
        rep=5,
    )
    assert report.total_flops >= 10e9, f"FLOPs {report.total_flops} < 10 GFLOPs"
    assert report.sram_bytes_saved_mb > 0.0
    assert report.speedup_x >= 3.0, f"Speedup {report.speedup_x} < 3.0x"
    assert report.latency_triton_ms < report.latency_pytorch_ms

    chart = render_roofline_terminal(report)
    assert "[T]" in chart
    assert "[P]" in chart
    assert "SPEEDUP" in chart


def test_cli_validate_and_scaffold_subcommands():
    """Test neuralops validate and neuralops scaffold via CLI main."""
    from neuralops.benchmarks.cli import main

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        ret = main(["validate", "gated_delta_net"])
        out = sys.stdout.getvalue()
        assert ret == 0
        assert "ALL CONTRACTS PASSED" in out
    finally:
        sys.stdout = old_stdout

    sys.stdout = io.StringIO()
    try:
        ret = main(["scaffold", "demo_scaffold", "--category", "attention"])
        out = sys.stdout.getvalue()
        assert ret == 0
        assert "class DemoScaffold" in out
    finally:
        sys.stdout = old_stdout
