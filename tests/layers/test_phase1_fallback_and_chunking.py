#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Regression suite for Phase 1 native-kernel fallback guards, power-of-2 guards,
dtype consistency, and memory-bounded chunked scans across:
  - hla_attention
  - hgrn2_gated_deltanet
  - hgrn2_gated_outer
  - hgrn2_gated_layer
  - mamba_block
  - mamba_svd_flash
  - selective_ssm
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.layers.gated.hgrn2_gated_deltanet import HGRN2GatedDeltaNet
from neuralops.layers.gated.hgrn2_gated_layer import HGRN2Layer
from neuralops.layers.gated.hgrn2_gated_outer import HGRN2GatedOuterBlock
from neuralops.layers.gated.hla_attention import HLAAttention
from neuralops.layers.ssm.mamba_block import MambaBlock
from neuralops.layers.ssm.mamba_svd_flash import MambaSVDFlash, inject_mamba_svd_flash
from neuralops.layers.ssm.selective_ssm import SelectiveSSM, stable_zoh_b

ALL_MODULES = [
    ("hla_attention", HLAAttention),
    ("hgrn2_gated_deltanet", HGRN2GatedDeltaNet),
    ("hgrn2_gated_outer", HGRN2GatedOuterBlock),
    ("hgrn2_gated_layer", HGRN2Layer),
    ("mamba_block", MambaBlock),
    ("mamba_svd_flash", MambaSVDFlash),
    ("selective_ssm", SelectiveSSM),
]

PROFILES = [
    ("BERT_768", dict(B=2, L=64, D=768, H=12, H_kv=12, is_decoder=False, mt="bert")),
    ("Qwen_896", dict(B=2, L=64, D=896, H=14, H_kv=2, is_decoder=True, mt="qwen2")),
    ("Qwen_L135", dict(B=1, L=135, D=896, H=14, H_kv=2, is_decoder=True, mt="qwen2")),
    ("LLaVA_1024", dict(B=1, L=64, D=1024, H=16, H_kv=16, is_decoder=True, mt="llama")),
    ("TinyLlama_2048_GQA", dict(B=1, L=64, D=2048, H=32, H_kv=4, is_decoder=True, mt="llama")),
]


def _make_module(cls, profile: dict):
    cfg = SimpleNamespace(
        hidden_size=profile["D"],
        num_attention_heads=profile["H"],
        num_key_value_heads=profile["H_kv"],
        head_dim=profile["D"] // profile["H"],
        is_decoder=profile["is_decoder"],
        is_causal=profile["is_decoder"],
        model_type=profile["mt"],
        lora_rank=8,
        lora_alpha=16.0,
    )
    try:
        return cls(config=cfg)
    except Exception:
        return cls(config=cfg, d_model=profile["D"], num_heads=profile["H"])


@pytest.mark.parametrize("name,cls", ALL_MODULES, ids=[m[0] for m in ALL_MODULES])
@pytest.mark.parametrize("pname,profile", PROFILES, ids=[p[0] for p in PROFILES])
def test_forward_backward_and_causal_contract(name, cls, pname, profile):
    torch.manual_seed(42)
    mod = _make_module(cls, profile)
    mod.train()

    B, L, D = profile["B"], profile["L"], profile["D"]
    x = torch.randn(B, L, D, requires_grad=True)
    mask = torch.ones(B, L, dtype=torch.long)
    if not profile["is_decoder"]:
        mask[:, -8:] = 0

    res = mod(x, attention_mask=mask)
    out = res[0] if isinstance(res, tuple) else res

    assert out.shape == (B, L, D)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, x, atol=1e-4)

    loss = (out**2).mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    trainable = [p for p in mod.parameters() if p.requires_grad]
    nonzero = sum(1 for p in trainable if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
    assert nonzero / max(1, len(trainable)) >= 0.80

    if profile["is_decoder"] and pname == "Qwen_896":
        mod.eval()
        with torch.no_grad():
            x1 = torch.randn(1, 32, D)
            x2 = x1.clone()
            x2[:, -1, :] += 5.0
            y1 = mod(x1)[0]
            y2 = mod(x2)[0]
            assert (y1[:, :-1] - y2[:, :-1]).abs().max().item() < 1e-5


def test_chunked_scans_match_step_by_step_reference():
    torch.manual_seed(7)

    # 1. HGRN2GatedOuterBlock._chunked_scan
    outer = HGRN2GatedOuterBlock(dim=128, num_heads=4, is_decoder=True)
    q = torch.randn(2, 4, 70, 32)
    k = torch.randn(2, 4, 70, 32)
    v = torch.randn(2, 4, 70, 32)
    f = torch.sigmoid(torch.randn(2, 4, 70)).clamp(0.05, 0.99)
    y_c, s_c = outer._chunked_scan(q, k, v, f, chunk_size=16)

    s_ref = torch.zeros(2, 4, 32, 32)
    ys = []
    for t in range(70):
        s_ref = s_ref * f[:, :, t].unsqueeze(-1).unsqueeze(-1) + torch.matmul(
            k[:, :, t].unsqueeze(-1), v[:, :, t].unsqueeze(-2)
        )
        ys.append(torch.matmul(q[:, :, t].unsqueeze(-2), s_ref).squeeze(-2).unsqueeze(2))
    y_ref = torch.cat(ys, dim=2)
    assert (y_c - y_ref).abs().max().item() < 1e-4
    assert (s_c - s_ref).abs().max().item() < 1e-4

    # 2. HGRN2Layer._scan_direction
    hgrn_layer = HGRN2Layer(d_model=128, is_decoder=True)
    f2 = torch.sigmoid(torch.randn(2, 70, 128)).clamp(0.05, 0.99)
    iv2 = torch.randn(2, 70, 128)
    h_c, h_last = hgrn_layer._scan_direction(f2, iv2, chunk_size=16)
    h_ref = torch.zeros(2, 128)
    hs = []
    for t in range(70):
        h_ref = f2[:, t] * h_ref + iv2[:, t]
        hs.append(h_ref)
    assert (h_c - torch.stack(hs, dim=1)).abs().max().item() < 1e-4
    assert (h_last - h_ref).abs().max().item() < 1e-4

    # 3. MambaSVDFlash._ssm_scan_from_module vs full _ssm_scan
    msf = MambaSVDFlash(d_model=64, num_heads=4, is_decoder=True)
    x3 = torch.randn(2, 70, 64)
    y3_mod, h3_mod = msf._ssm_scan_from_module(msf.ssm_fwd, x3, chunk_size=32)
    u3, sig3, v3, b3_s, c3_s = msf.ssm_fwd(x3)
    y3_ref, h3_ref = msf._ssm_scan(u3, sig3, v3, b3_s, c3_s, x3)
    assert (y3_mod - y3_ref).abs().max().item() < 1e-4
    assert (h3_mod - h3_ref).abs().max().item() < 1e-4

    # 4. SelectiveSSM._chunked_pytorch_scan (no double-softplus)
    sssm = SelectiveSSM(d_model=64, d_state=16, is_decoder=True)
    u4 = torch.randn(2, 128, 70)
    dt4 = F.softplus(torch.randn(2, 128, 70))
    a4 = -torch.exp(sssm.A_log.detach())
    b4 = torch.randn(2, 16, 70)
    c4 = torch.randn(2, 16, 70)
    d4 = sssm.D.detach()
    y4_c, h4_c = sssm._chunked_pytorch_scan(u4, dt4, a4, b4, c4, d4, chunk_size=32)
    h4 = torch.zeros(2, 128, 16)
    ys4 = []
    for t in range(70):
        da = (dt4[:, :, t].unsqueeze(-1) * a4.unsqueeze(0)).clamp(-20.0, 0.0)
        h4 = torch.exp(da) * h4 + stable_zoh_b(da) * dt4[:, :, t].unsqueeze(-1) * (
            u4[:, :, t].unsqueeze(-1) * b4[:, :, t].unsqueeze(1)
        )
        ys4.append((h4 * c4[:, :, t].unsqueeze(1)).sum(-1) + d4.unsqueeze(0) * u4[:, :, t])
    assert (y4_c - torch.stack(ys4, dim=-1)).abs().max().item() < 1e-4
    assert (h4_c - h4).abs().max().item() < 1e-4

    # 5. MambaBlock._recurrent_scan
    mb = MambaBlock(d_model=64, d_state=16, is_decoder_override=True)
    y5_c, h5_c = mb._recurrent_scan(u4, dt4, a4, b4, c4, d4, None, chunk_size=32)
    dt5_sp = F.softplus(dt4)
    h5 = torch.zeros(2, 128, 16)
    ys5 = []
    for t in range(70):
        da = (dt5_sp[:, :, t].unsqueeze(-1) * a4.unsqueeze(0)).clamp(-20.0, 0.0)
        h5 = torch.exp(da) * h5 + dt5_sp[:, :, t].unsqueeze(-1) * b4[:, :, t].unsqueeze(1) * u4[:, :, t].unsqueeze(-1)
        ys5.append((h5 * c4[:, :, t].unsqueeze(1)).sum(-1) + d4.unsqueeze(0) * u4[:, :, t])
    assert (y5_c - torch.stack(ys5, dim=-1)).abs().max().item() < 1e-4
    assert (h5_c - h5).abs().max().item() < 1e-4


def test_hgrn2_layer_non_pow2_skips_kernel_and_chunks_local_attn(caplog):
    # D=896 is non-power-of-2; Triton scan must NOT be called
    layer = HGRN2Layer(d_model=896, is_decoder=True, window_size=32)
    x = torch.randn(1, 160, 896, requires_grad=True)
    with patch("neuralops.layers.gated.hgrn2_gated_layer.hgrn2_scan_triton") as mock_triton:
        out, _ = layer(x)
        mock_triton.assert_not_called()
    assert out.shape == (1, 160, 896)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # Check chunked _local_attention produces valid finite output when L > chunk_q (64)
    layer.eval()
    with torch.no_grad():
        out_chunked, _, _ = layer._local_attention(x)
        assert out_chunked.shape == (1, 160, 896)
        assert torch.isfinite(out_chunked).all()

    # Power-of-2 D with simulated accelerator tensor + failing kernel emits warning & falls back cleanly
    layer_p2 = HGRN2Layer(d_model=64, is_decoder=True)
    f = torch.full((1, 16, 64), 0.9)
    iv = torch.randn(1, 16, 64)
    with (
        caplog.at_level(logging.WARNING),
        patch.object(torch.Tensor, "is_cuda", new=True),
        patch("neuralops.layers.gated.hgrn2_gated_layer.HAS_TRITON_HGRN2", True),
        patch("neuralops.layers.gated.hgrn2_gated_layer.hgrn2_scan_triton", side_effect=RuntimeError("mock kernel err")),
    ):
        h_out, _ = layer_p2._scan_direction(f, iv)
    assert any("falling back to chunked PyTorch" in r.message for r in caplog.records)
    assert torch.isfinite(h_out).all()


def test_simulated_compiler_failure_fallback_logs_warnings(caplog):
    # 1. MambaBlock: _get_cuda_ss_module returns None -> logs warning & falls back to _recurrent_scan
    mb = MambaBlock(d_model=64, d_state=16, is_decoder_override=True)
    u = torch.randn(1, 32, 16)
    dt = torch.randn(1, 32, 16)
    a = -torch.ones(32, 16)
    b = torch.randn(1, 16, 16)
    c = torch.randn(1, 16, 16)
    d = torch.ones(32)
    with (
        caplog.at_level(logging.WARNING),
        patch("neuralops.layers.ssm.mamba_block._WARNED_CUDA_SS_UNAVAILABLE", False),
        patch("neuralops.layers.ssm.mamba_block._get_cuda_ss_module", return_value=None),
        patch("neuralops.layers.ssm.mamba_block.HAS_TRITON_MAMBA", False),
        patch.object(torch.Tensor, "is_cuda", new=True),
    ):
        y = mb._parallel_scan(u, dt, a, b, c, d)
    assert any("CUDA selective_scan extension unavailable" in r.message for r in caplog.records)
    assert y.shape == u.shape and torch.isfinite(y).all()

    caplog.clear()

    # 2. SelectiveSSM: _get_cuda_ss_module returns None -> logs warning & falls back to _chunked_pytorch_scan
    sssm = SelectiveSSM(d_model=64, d_state=16, is_decoder=True)
    x_ssm = torch.randn(1, 16, 128, requires_grad=True)
    z = torch.randn(1, 16, 128, requires_grad=True)
    with (
        caplog.at_level(logging.WARNING),
        patch("neuralops.layers.ssm.selective_ssm._WARNED_CUDA_SS_UNAVAILABLE", False),
        patch("neuralops.layers.ssm.selective_ssm._get_cuda_ss_module", return_value=None),
        patch("neuralops.layers.ssm.selective_ssm.triton_selective_scan", None),
        patch.object(torch.Tensor, "is_cuda", new=True),
    ):
        y_core, _ = sssm._forward_core(
            x_ssm,
            z,
            sssm.conv1d,
            sssm.x_proj,
            sssm.dt_proj,
            sssm.A_log,
            sssm.D,
        )
    assert any("CUDA selective_scan extension unavailable" in r.message for r in caplog.records)
    assert y_core.shape == x_ssm.shape and torch.isfinite(y_core).all()

    caplog.clear()

    # 3. HLAAttention: _get_module returns None -> logs warning & falls back cleanly
    hla = HLAAttention(d_model=64, num_heads=4, is_decoder=True)
    q = torch.rand(1, 4, 16, 16) + 0.5
    k = torch.rand(1, 4, 16, 16) + 0.5
    v = torch.randn(1, 4, 16, 16)
    decay = torch.full((1, 4, 1, 1), 0.9)
    with (
        caplog.at_level(logging.WARNING),
        patch("neuralops.ops.cuda.hla_attention.hla_attention_autograd._get_module", return_value=None),
        patch("neuralops.ops.triton.gla_scan.kda_hla_chunk_scan_triton", side_effect=RuntimeError("mock kernel")),
        patch.object(torch.Tensor, "is_cuda", new=True),
    ):
        y_hla, _ = hla._hla_chunked_scan(q, k, v, decay)
    assert any("cuda_hla_attention unavailable" in r.message for r in caplog.records)
    assert any("kda_hla_chunk_scan_triton failed" in r.message for r in caplog.records)
    assert y_hla.shape == q.shape and torch.isfinite(y_hla).all()


def test_hla_gqa_step_and_mixed_dtype_safety():
    cfg = SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        is_decoder=True,
        model_type="qwen2",
    )
    hla = HLAAttention(config=cfg)
    state = None
    for _ in range(3):
        x_step = torch.randn(2, 1, 64)
        y_step, state = hla.step(x_step, state=state)
        assert y_step.shape == (2, 1, 64)
        assert torch.isfinite(y_step).all()

    # Mixed dtype input (bfloat16 input with float32 module weights without autocast)
    for _, cls in ALL_MODULES:
        mod = _make_module(cls, dict(B=1, L=16, D=64, H=4, H_kv=2, is_decoder=True, mt="qwen2"))
        x_bf16 = torch.randn(1, 16, 64, dtype=torch.bfloat16, requires_grad=True)
        res = mod(x_bf16)
        out = res[0] if isinstance(res, tuple) else res
        assert out.shape == (1, 16, 64)
        assert torch.isfinite(out).all()
        out.float().sum().backward()
        assert x_bf16.grad is not None and torch.isfinite(x_bf16.grad).all()


def test_inject_mamba_svd_flash_preserves_dtype():
    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)
            self.hidden_size = 64
            self.num_heads = 4

    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummyAttn().to(dtype=torch.bfloat16)

    layer = DummyLayer()
    cfg = SimpleNamespace(hidden_size=64, num_attention_heads=4, is_decoder=True, model_type="qwen2")
    injected = inject_mamba_svd_flash(layer=layer, model_config=cfg, lora_rank=8)
    assert injected is not None
    assert next(injected.parameters()).dtype == torch.bfloat16
    x = torch.randn(1, 12, 64, dtype=torch.bfloat16)
    out, _ = layer.self_attn(x)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
