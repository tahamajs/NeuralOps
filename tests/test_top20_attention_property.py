# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Automated property-based and autograd verification for Top-20 Attention Mechanisms.
Tests device/dtype preservation (FP32, FP16, BF16), autograd stability, and high GPU saturation.
"""

import pytest
import torch
from neuralops.core.registry import ATTENTION_REGISTRY

TOP_20_ATTENTIONS = [
    "flash_attention3",
    "flash_attention3_fp8",
    "cosformer",
    "gla_attention",
    "gated_delta_net",
    "gated_deltanet_block",
    "mla_attention",
    "cluster_attention",
    "routing_attention",
    "adaptive_sparse_attention",
    "mamba_block",
    "hgrn2_gated_deltanet",
    "spectraformer_attention",
    "primal_attention",
    "linformer_attention",
    "performer_attention",
    "nystrom_attention",
    "gqa_attention",
    "sink_attention",
    "svd_linear_attention",
]


@pytest.fixture(scope="session", autouse=True)
def setup_cuda_env():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


@pytest.mark.parametrize("attn_name", TOP_20_ATTENTIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_attention_device_dtype_and_autograd(attn_name, dtype):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("Half precision skipped on CPU")

    cls = ATTENTION_REGISTRY.get(attn_name)
    assert cls is not None, f"Module {attn_name} not found in ATTENTION_REGISTRY"

    B, L, D = 2, 64, 768
    mod = cls().to(device=device, dtype=dtype)
    x = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=True)

    out = mod(x)
    if isinstance(out, tuple):
        out = out[0]

    assert out.shape == (B, L, D), f"Expected shape {(B, L, D)}, got {out.shape}"
    assert out.dtype == dtype, f"Expected dtype {dtype}, got {out.dtype}"
    assert out.device.type == device, f"Expected device {device}, got {out.device.type}"

    loss = out.sum()
    loss.backward()

    assert x.grad is not None, f"Gradient is None for {attn_name} {dtype}"
    assert not torch.isnan(x.grad).any(), f"NaN in gradient for {attn_name} {dtype}"
    assert not torch.isinf(x.grad).any(), f"Inf in gradient for {attn_name} {dtype}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for saturation test")
@pytest.mark.parametrize("attn_name", TOP_20_ATTENTIONS)
def test_attention_high_saturation_bf16(attn_name):
    cls = ATTENTION_REGISTRY.get(attn_name)
    B, L, D = 8, 512, 768
    mod = cls().to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(B, L, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = mod(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()

    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert out.shape == (B, L, D)
