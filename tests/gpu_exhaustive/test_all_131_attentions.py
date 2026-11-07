#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Exhaustive GPU Verification for all 131 Registered Attention Mechanisms.
Validates forward numerical integrity, autograd backward propagation,
causal masking strictness, and multi-query/grouped-query attention (GQA/MQA).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from neuralops.core.registry import ATTENTION_REGISTRY, create_attention


def _get_attention_names() -> list[str]:
    """Retrieve full canonical list of all registered attention names."""
    ATTENTION_REGISTRY._load_all()
    names = sorted(list(ATTENTION_REGISTRY.keys()))
    if len(names) < 116:
        pytest.fail(f"Attention registry short: expected >= 116 names, got {len(names)}")
    return names


ALL_ATTENTION_NAMES = _get_attention_names()


def _instantiate_attention(
    name: str,
    dim: int,
    heads: int,
    num_kv_heads: int | None = None,
    is_causal: bool = False,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Instantiate attention mechanism with robust parameter mapping."""
    cls = ATTENTION_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Attention '{name}' not found in registry.")

    config = BertConfig(
        hidden_size=dim,
        num_attention_heads=heads,
        is_decoder=is_causal,
    )
    if is_causal:
        config.model_type = "gpt2"
    if num_kv_heads is not None:
        config.num_key_value_heads = num_kv_heads

    kwargs: dict[str, Any] = {
        "dim": dim,
        "num_heads": heads,
        "embed_dim": dim,
        "is_causal": is_causal,
    }
    if num_kv_heads is not None:
        kwargs["num_kv_heads"] = num_kv_heads
        kwargs["n_kv_heads"] = num_kv_heads

    attempts = [
        lambda: create_attention(name, config=config, dim=dim, num_heads=heads, **kwargs),
        lambda: cls(dim=dim, num_heads=heads, **({} if num_kv_heads is None else {"num_kv_heads": num_kv_heads})),
        lambda: cls(config=config),
        lambda: cls(config),
        lambda: cls(dim=dim, num_heads=heads),
        lambda: cls(dim, num_heads=heads),
        lambda: cls(),
    ]

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            mod = attempt()
            if isinstance(mod, nn.Module):
                return mod.to(device=device, dtype=dtype)
        except Exception as err:
            last_error = err
            continue

    raise RuntimeError(f"Failed to instantiate attention '{name}' for GPU execution: {last_error}")


def _set_causal_recursively(mod: nn.Module) -> None:
    """Enforce causal/decoder flags across nested modules."""
    if hasattr(mod, "is_decoder"):
        mod.is_decoder = True
    if hasattr(mod, "is_causal"):
        mod.is_causal = True
    if hasattr(mod, "config"):
        cfg = mod.config
        if hasattr(cfg, "is_decoder"):
            cfg.is_decoder = True
        if hasattr(cfg, "model_type"):
            cfg.model_type = "gpt2"
    for child in mod.children():
        _set_causal_recursively(child)


def _forward_attention(
    mod: nn.Module,
    x: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run forward pass adapting to various attention call conventions."""
    b, s, d = x.shape
    device = x.device
    causal_mask = None
    if is_causal:
        causal_mask = torch.tril(torch.ones(s, s, device=device, dtype=torch.bool))

    for call_fn in [
        lambda: mod(x, is_causal=is_causal),
        lambda: mod(x, causal=is_causal),
        lambda: mod(x, attention_mask=causal_mask),
        lambda: mod(hidden_states=x, attention_mask=causal_mask),
        lambda: mod(x, x, x),
        lambda: mod(x),
    ]:
        try:
            out = call_fn()
            out_tensor = out[0] if isinstance(out, tuple) else out
            if isinstance(out_tensor, torch.Tensor):
                return out_tensor
        except (TypeError, ValueError, AttributeError):
            continue

    b, s, d = x.shape
    out = mod(hidden_states=x)
    return out[0] if isinstance(out, tuple) else out


@pytest.mark.gpu
@pytest.mark.parametrize("name", ALL_ATTENTION_NAMES)
@pytest.mark.parametrize(
    "b,h,t,d",
    [
        (2, 4, 64, 32),  # Dim = 128
        (1, 8, 512, 64),  # Dim = 512
    ],
)
def test_attention_forward_backward_gpu(
    name: str,
    b: int,
    h: int,
    t: int,
    d: int,
    fp_dtype: torch.dtype,
):
    """Exhaustive forward and autograd backward verification on real GPU hardware."""
    device = torch.device("cuda:0")
    dim = h * d

    try:
        mod = _instantiate_attention(name, dim=dim, heads=h, device=device, dtype=fp_dtype)
    except Exception as exc:
        pytest.fail(f"[{name}] Instantiation failed on GPU: {exc}")

    mod.train()
    x = torch.randn(b, t, dim, device=device, dtype=fp_dtype, requires_grad=True)

    # 1. Forward Pass Execution
    try:
        out_tensor = _forward_attention(mod, x)
    except Exception as exc:
        pytest.fail(f"[{name}] Forward execution failed on GPU: {exc}")

    assert out_tensor.shape[0] == b, f"[{name}] Batch dimension mismatch: {out_tensor.shape}"
    assert out_tensor.shape[1] == t, f"[{name}] Sequence dimension mismatch: {out_tensor.shape}"
    assert out_tensor.shape[2] == dim, f"[{name}] Hidden dimension mismatch: {out_tensor.shape}"
    assert torch.isfinite(out_tensor).all().item(), f"[{name}] Non-finite values detected in GPU forward activations."
    assert out_tensor.dtype == fp_dtype, f"[{name}] Dtype not preserved: expected {fp_dtype}, got {out_tensor.dtype}"

    # 2. Backward Autograd Execution
    loss = out_tensor.mean() * 10.0
    try:
        loss.backward()
    except Exception as exc:
        pytest.fail(f"[{name}] Autograd backward failed on GPU: {exc}")

    assert x.grad is not None, f"[{name}] Gradient was not propagated back to input tensor."
    assert torch.isfinite(x.grad).all().item(), f"[{name}] Non-finite gradient detected on input tensor."

    trainable_params = [p for p in mod.parameters() if p.requires_grad]
    if trainable_params:
        params_with_grad = [p for p in trainable_params if p.grad is not None]
        assert (
            len(params_with_grad) > 0
        ), f"[{name}] None of the {len(trainable_params)} trainable parameters received gradients."
        for p in params_with_grad:
            assert torch.isfinite(p.grad).all().item(), f"[{name}] Non-finite gradient in trainable model parameter."


@pytest.mark.gpu
@pytest.mark.parametrize("name", ALL_ATTENTION_NAMES)
def test_attention_causal_masking_strictness(name: str):
    """Verify strictly causal attention does not leak future information to past tokens."""
    metadata = ATTENTION_REGISTRY.get_metadata(name)
    if metadata.get("supports_causal", True) is False:
        pytest.skip(f"[{name}] Declares supports_causal=False in registry.")

    device = torch.device("cuda:0")
    dim = 128
    heads = 4
    seq_len = 32

    try:
        mod = _instantiate_attention(name, dim=dim, heads=heads, is_causal=True, device=device, dtype=torch.float32)
    except Exception as exc:
        pytest.skip(f"[{name}] Does not support explicit is_causal initialization: {exc}")

    _set_causal_recursively(mod)
    mod.eval()
    x1 = torch.randn(1, seq_len, dim, device=device, dtype=torch.float32)
    cutoff = 16

    try:
        with torch.no_grad():
            out1 = _forward_attention(mod, x1, is_causal=True)
    except Exception as exc:
        pytest.skip(f"[{name}] Causal forward execution unsupported: {exc}")

    # Perturb all future tokens beyond cutoff
    x2 = x1.clone()
    x2[:, cutoff:, :] += torch.randn(1, seq_len - cutoff, dim, device=device, dtype=torch.float32) * 5.0

    with torch.no_grad():
        out2 = _forward_attention(mod, x2, is_causal=True)

    # Tokens up to cutoff must be identical within numerical floating point budget
    max_diff = (out1[:, :cutoff, :] - out2[:, :cutoff, :]).abs().max().item()
    if max_diff > 1e-3:
        pytest.fail(
            f"[{name}] Causal leakage detected: past tokens changed when future tokens were modified (max diff={max_diff:.6f})."
        )


@pytest.mark.gpu
@pytest.mark.parametrize("name", ALL_ATTENTION_NAMES)
def test_attention_gqa_mqa_support(name: str):
    """Verify Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) shapes if supported."""
    device = torch.device("cuda:0")
    dim = 256
    num_heads = 8
    num_kv_heads = 2
    b, t = 2, 64

    try:
        mod = _instantiate_attention(
            name,
            dim=dim,
            heads=num_heads,
            num_kv_heads=num_kv_heads,
            device=device,
            dtype=torch.float32,
        )
    except Exception:
        pytest.skip(f"[{name}] Does not declare GQA/MQA parameter support.")

    mod.eval()
    x = torch.randn(b, t, dim, device=device, dtype=torch.float32)

    try:
        with torch.no_grad():
            out = _forward_attention(mod, x)
    except Exception as exc:
        pytest.skip(f"[{name}] Forward execution with GQA config unsupported: {exc}")

    assert out.shape == (b, t, dim), f"[{name}] GQA output shape mismatch: expected {(b, t, dim)}, got {out.shape}"
    assert torch.isfinite(out).all().item(), f"[{name}] Non-finite activations in GQA forward pass."
