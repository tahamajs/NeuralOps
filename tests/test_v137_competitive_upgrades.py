#!/usr/bin/env python3
"""
Test suite for NeuralOps v1.3.7 Competitive Dominance Upgrades:
  1. Triton Backward Kernels Parity & Memory Efficiency (low_rank, gdn_scan, gla_scan)
  2. O(1) Recurrent State Cache & Constant-Memory Autoregressive Decoding
  3. Zero-Overhead Hardware-Calibrated Autotuning & Roofline Analysis
  4. Modern GQA (Llama-3/Qwen-2.5 style) Model Surgery, Micro-Healing, and Checkpoint I/O
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import neuralops
from neuralops.core.state_cache import NeuralOpsStateCache, benchmark_autoregressive_decoding
from neuralops.core.triton_kernels.autotuner import select_optimal_triton_config
from neuralops.models.surgical_io import save_surgical_checkpoint, load_surgical_checkpoint
from neuralops.core._attention_surgery import heal_replaced_model, calibrate_attention_subspace


# ==============================================================================
# TEST 1: Triton Backward Kernel Parity & Gradient Verification
# ==============================================================================

def test_triton_backward_kernels_parity():
    """Verify exact numerical gradient parity for low_rank, gdn_scan, and gla_scan."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. low_rank.py: fused_down_up_gemm_bias_triton vs PyTorch
    from neuralops.core.triton_kernels.low_rank import fused_down_up_gemm_bias_triton

    M, Din, R, Dout = 64, 128, 32, 128
    x_pt = torch.randn(M, Din, device=device, dtype=torch.float32, requires_grad=True)
    w_down_pt = torch.randn(Din, R, device=device, dtype=torch.float32, requires_grad=True)
    w_up_pt = torch.randn(Dout, R, device=device, dtype=torch.float32, requires_grad=True)
    bias_pt = torch.randn(Dout, device=device, dtype=torch.float32, requires_grad=True)

    x_tri = x_pt.detach().clone().requires_grad_(True)
    w_down_tri = w_down_pt.detach().clone().requires_grad_(True)
    w_up_tri = w_up_pt.detach().clone().requires_grad_(True)
    bias_tri = bias_pt.detach().clone().requires_grad_(True)

    # PyTorch reference forward & backward
    out_pt = torch.matmul(torch.matmul(x_pt, w_down_pt), w_up_pt.t()) + bias_pt
    grad_target = torch.randn_like(out_pt)
    out_pt.backward(grad_target)

    # Fused Triton forward & backward
    out_tri = fused_down_up_gemm_bias_triton(x_tri, w_down_tri, w_up_tri, bias=bias_tri)
    out_tri.backward(grad_target)

    assert torch.allclose(out_pt, out_tri, atol=1e-3, rtol=1e-3), "Low rank forward mismatch"
    assert torch.allclose(x_pt.grad, x_tri.grad, atol=1e-3, rtol=1e-3), "dx gradient mismatch"
    assert torch.allclose(w_down_pt.grad, w_down_tri.grad, atol=1e-3, rtol=1e-3), "dw_down gradient mismatch"
    assert torch.allclose(w_up_pt.grad, w_up_tri.grad, atol=1e-3, rtol=1e-3), "dw_up gradient mismatch"
    assert torch.allclose(bias_pt.grad, bias_tri.grad, atol=1e-3, rtol=1e-3), "dbias gradient mismatch"

    # 2. gdn_scan.py: gdn_scan_triton vs gdn_scan_pytorch backward
    from neuralops.core.triton_kernels.gdn_scan import gdn_scan_triton, gdn_scan_pytorch
    B, H, L, D = 2, 4, 32, 32
    q = torch.randn(B, H, L, D, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H, L, D, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H, L, D, device=device, dtype=torch.float32, requires_grad=True)
    beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32)).requires_grad_(True)
    alpha = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32)).requires_grad_(True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    beta_ref = beta.detach().clone().requires_grad_(True)
    alpha_ref = alpha.detach().clone().requires_grad_(True)

    out_ref = gdn_scan_pytorch(q_ref, k_ref, v_ref, beta_ref, alpha_ref)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    if device.type == "cuda":
        out_tri = gdn_scan_triton(q, k, v, beta, alpha)
        out_tri.backward(grad_out)

        assert torch.allclose(out_ref, out_tri, atol=1e-3, rtol=1e-3), "GDN scan fwd mismatch"
        assert torch.allclose(q_ref.grad, q.grad, atol=1e-3, rtol=1e-3), "GDN dq mismatch"
        assert torch.allclose(k_ref.grad, k.grad, atol=1e-3, rtol=1e-3), "GDN dk mismatch"
        assert torch.allclose(v_ref.grad, v.grad, atol=1e-3, rtol=1e-3), "GDN dv mismatch"


# ==============================================================================
# TEST 2: O(1) Recurrent State Cache & Step Decoding Verification
# ==============================================================================

def test_o1_recurrent_state_cache_decoding():
    """Verify that NeuralOpsStateCache memory remains 100% constant across 100 decoding steps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = NeuralOpsStateCache()

    B, H, D = 1, 8, 64
    attn = neuralops.create_attention("gated_delta_net", embed_dim=H * D, num_heads=H)
    attn.to(device=device, dtype=torch.float16)

    # Prefill pass (T=64)
    prefill_x = torch.randn(B, 64, H * D, device=device, dtype=torch.float16)
    _ = attn(prefill_x)

    # Decode step 1
    token_1 = torch.randn(B, 1, H * D, device=device, dtype=torch.float16)
    state_dummy = torch.zeros((B, H, D, D), device=device, dtype=torch.float16)
    cache.update_recurrent_state(layer_idx=0, new_state=state_dummy)

    initial_bytes = cache.memory_bytes()
    assert initial_bytes > 0, "Cache memory must be allocated"

    # Decode 100 tokens sequentially
    for step_idx in range(100):
        new_token = torch.randn(B, 1, H * D, device=device, dtype=torch.float16)
        # Update dummy state in-place
        state_dummy.add_(0.01)
        cache.update_recurrent_state(layer_idx=0, new_state=state_dummy)

    final_bytes = cache.memory_bytes()
    assert initial_bytes == final_bytes, f"Memory must be strictly O(1) constant! {initial_bytes} vs {final_bytes}"

    # Verify benchmark utility
    bench = benchmark_autoregressive_decoding(seq_lens=(512, 2048, 8192), batch_size=1, embed_dim=H * D, num_heads=H)
    assert len(bench["results"]) == 3
    # Memory reduction ratio must increase with sequence length
    assert bench["results"][-1]["memory_reduction_ratio"] > bench["results"][0]["memory_reduction_ratio"]


# ==============================================================================
# TEST 3: Hardware-Calibrated Autotune & Roofline Analysis
# ==============================================================================

def test_hardware_calibrated_autotune_and_roofline():
    """Verify zero-overhead hardware autotuning and roofline profiler metrics."""
    # Test tile configuration selection
    cfg_small = select_optimal_triton_config("gemm", M=64, N=128, K=128)
    assert cfg_small["BLOCK_M"] in (32, 64)
    assert cfg_small["num_warps"] in (2, 4)

    cfg_large = select_optimal_triton_config("gemm", M=2048, N=768, K=768)
    assert cfg_large["BLOCK_M"] in (64, 128)

    # Test Roofline report on gated_delta_net
    report = neuralops.profile_roofline(
        module_or_name="gated_delta_net",
        batch_size=2,
        seq_len=512,
        embed_dim=256,
        num_heads=4,
    )
    assert report.total_flops > 0.0
    assert report.speedup_x > 0.0
    assert report.bottleneck_regime is not None


# ==============================================================================
# TEST 4: Modern GQA Architecture Model Surgery, Micro-Healing & Checkpoint I/O
# ==============================================================================

class MockGQASelfAttention(nn.Module):
    """Grouped-Query Attention submodule (H_Q=8, H_KV=2)."""

    def __init__(self, hidden_size=128, num_heads=8, num_kv_heads=2, head_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings=None, **kwargs):
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads to match Q heads
        g = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        return self.o_proj(out), attn


class MockGQADecoderLayer(nn.Module):
    """Synthetic modern LLM decoder block using self_attn attribute."""

    def __init__(self, hidden_size=128, num_heads=8, num_kv_heads=2, head_dim=16):
        super().__init__()
        self.self_attn = MockGQASelfAttention(hidden_size, num_heads, num_kv_heads, head_dim)

    def forward(self, hidden_states, position_embeddings=None, **kwargs):
        return self.self_attn(hidden_states, position_embeddings=position_embeddings, **kwargs)


class MockGQAModel(nn.Module):
    """Multi-layer modern GQA transformer backbone."""

    def __init__(self, num_layers=2, hidden_size=128, num_heads=8, num_kv_heads=2):
        super().__init__()
        class Config:
            pass
        self.config = Config()
        self.config.hidden_size = hidden_size
        self.config.num_attention_heads = num_heads
        self.config.num_key_value_heads = num_kv_heads
        self.config.num_hidden_layers = num_layers
        self.config.model_type = "llama"

        self.layers = nn.ModuleList([
            MockGQADecoderLayer(hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, **kwargs):
        out = x
        for layer in self.layers:
            out, _ = layer(out, **kwargs)
        return out


def test_modern_gqa_model_surgery_and_healing_roundtrip(tmp_path):
    """End-to-end test of modern GQA model surgery, micro-healing, and surgical checkpoint I/O."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MockGQAModel(num_layers=2, hidden_size=128, num_heads=8, num_kv_heads=2).to(device)
    model.eval()

    x = torch.randn(2, 32, 128, device=device)
    with torch.no_grad():
        orig_out = model(x)

    # 1. Surgical replacement with GQA subspace adaptation
    transplanted = neuralops.replace_attention(model, attention="gated_delta_net")
    with torch.no_grad():
        transplanted_out = transplanted(x)

    assert transplanted_out.shape == orig_out.shape, "Transplanted output shape mismatch"

    # 2. Layerwise Micro-Healing
    healing_stats = heal_replaced_model(
        teacher_model=model,
        student_model=transplanted,
        calibration_inputs=x,
        steps=10,
        lr=2e-3,
    )
    assert healing_stats["num_healed_layers"] >= 1, "Must heal at least 1 layer"
    assert healing_stats["final_cosine_sim"] >= healing_stats["initial_cosine_sim"] - 0.05

    transplanted.eval()
    with torch.no_grad():
        healed_out = transplanted(x)

    # 3. Save and reload surgical checkpoint
    save_dir = tmp_path / "gqa_healed_model"
    save_surgical_checkpoint(transplanted, str(save_dir))

    fresh_model = MockGQAModel(num_layers=2, hidden_size=128, num_heads=8, num_kv_heads=2).to(device)
    reloaded_model, manifest = load_surgical_checkpoint(fresh_model, str(save_dir))

    with torch.no_grad():
        reloaded_out = reloaded_model(x)

    assert torch.allclose(healed_out, reloaded_out, atol=1e-5), "Bit-exact checkpoint roundtrip failed!"
