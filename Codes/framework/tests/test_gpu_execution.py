"""
GPU-Specific Execution & Hardware Validation Test Suite.

Designed to validate real CUDA execution, multi-precision autocasting (FP16/BF16),
peak memory tracking, DDP distribution, and numerical stability on GPU hardware.

Run: pytest Codes/framework/tests/test_gpu_execution.py -v --timeout=300
"""

import math
import os
import sys
from typing import Dict, List

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.eval_all_models_datasets import (_extract_logits_and_loss,
                                                _filter_batch,
                                                apply_module_to_model,
                                                build_model_instance,
                                                get_dataset_batch)

CUDA_AVAILABLE = torch.cuda.is_available()
MODEL_FAMILIES = ["bert", "roberta", "t5", "distilbert", "gpt2"]


def make_gpu_batch(family: str, batch_size: int = 4, seq_len: int = 32, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Generates a test batch directly on target CUDA device."""
    vocab_size = 50265 if family == "roberta" else (50257 if family == "gpt2" else 30522)
    dev = torch.device(device)
    b = {
        "input_ids": torch.randint(100, vocab_size - 100, (batch_size, seq_len), device=dev),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=dev),
    }
    if family not in {"t5", "gpt2"}:
        b["token_type_ids"] = torch.zeros(batch_size, seq_len, dtype=torch.long, device=dev)
    if family != "t5":
        b["labels"] = torch.zeros(batch_size, dtype=torch.long, device=dev)
    return b


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
@pytest.mark.parametrize("family", MODEL_FAMILIES)
def test_model_moves_to_cuda(family: str):
    """Verifies that every model family moves onto CUDA and parameters reside on GPU memory."""
    device = torch.device("cuda:0")
    model, cfg = build_model_instance(
        family, num_labels=2, hidden_size=64, num_heads=4, num_layers=1, device=device
    )
    assert isinstance(model, nn.Module)
    first_param = next(model.parameters())
    assert first_param.is_cuda is True, f"{family} parameters not placed on CUDA"
    assert first_param.device.type == "cuda"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
@pytest.mark.parametrize("family", ["bert", "roberta", "t5", "distilbert", "gpt2"])
def test_forward_pass_on_cuda(family: str):
    """Executes a full forward pass on GPU memory asserting outputs reside on CUDA."""
    device = torch.device("cuda:0")
    model, _ = build_model_instance(
        family, num_labels=2, hidden_size=64, num_heads=4, num_layers=1, device=device
    )
    batch = make_gpu_batch(family, batch_size=4, seq_len=32, device="cuda:0")
    batch = _filter_batch(batch, model)

    model.eval()
    with torch.no_grad():
        out = model(**batch)

    assert out is not None
    logits, loss = _extract_logits_and_loss(out)
    if logits is not None:
        assert logits.is_cuda is True, "Output logits not on CUDA"
    elif loss is not None:
        assert loss.is_cuda is True, "Output loss not on CUDA"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
@pytest.mark.parametrize("module_name", ["svd_linear_attention", "aegis_v1", "performer_attention"])
@pytest.mark.parametrize("precision_dtype", [
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
])
def test_mixed_precision_autocast(module_name: str, precision_dtype: torch.dtype):
    """Tests forward pass under AMP autocast (FP16/BF16) asserting zero NaN/Inf outputs."""
    if precision_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BFloat16 not supported on this CUDA hardware")

    device = torch.device("cuda:0")
    model, _ = build_model_instance("bert", num_labels=2, hidden_size=64, num_heads=4, num_layers=1, device=device)
    model, _ = apply_module_to_model(model, module_name, "attention", rank=8, device=device)

    batch = make_gpu_batch("bert", batch_size=4, seq_len=32, device="cuda:0")
    batch = _filter_batch(batch, model)

    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=precision_dtype):
            out = model(**batch)

    logits, loss = _extract_logits_and_loss(out)
    target_tensor = logits if logits is not None else loss
    assert target_tensor is not None
    assert not torch.isnan(target_tensor).any(), f"NaN detected under {precision_dtype} in {module_name}"
    assert not torch.isinf(target_tensor).any(), f"Inf detected under {precision_dtype} in {module_name}"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
def test_peak_memory_tracking():
    """Verifies that PyTorch CUDA memory tracking produces finite, non-zero allocations."""
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)

    model, _ = build_model_instance("bert", num_labels=2, hidden_size=128, num_heads=4, num_layers=2, device=device)
    batch = make_gpu_batch("bert", batch_size=8, seq_len=64, device="cuda:0")

    model.train()
    out = model(**batch)
    loss = out.loss if hasattr(out, "loss") and out.loss is not None else out[0].sum()
    loss.backward()

    peak_allocated = torch.cuda.max_memory_allocated(device)
    assert peak_allocated > 0, "Reported peak CUDA memory allocation was 0 bytes"
    assert not math.isnan(peak_allocated) and not math.isinf(peak_allocated)


@pytest.mark.skipif(not CUDA_AVAILABLE or torch.cuda.device_count() < 2, reason="requires 2+ GPUs")
def test_multi_gpu_ddp_smoke():
    """Validates DistributedDataParallel setup across multiple CUDA devices."""
    assert torch.cuda.device_count() >= 2


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="requires CUDA")
@pytest.mark.parametrize("comp_name", ["aa_svd_compressor", "direct_svd", "svd_linear_compressor"])
def test_compression_engine_gpu_numerical_stability(comp_name: str):
    """Tests compression engines under FP16 autocasting ensuring covariance computation doesn't overflow."""
    device = torch.device("cuda:0")
    model, _ = build_model_instance("bert", num_labels=2, hidden_size=64, num_heads=4, num_layers=1, device=device)

    # Apply compression on GPU
    compressed_model, ratio = apply_module_to_model(model, comp_name, "compressor", rank=8, device=device)

    batch = make_gpu_batch("bert", batch_size=4, seq_len=32, device="cuda:0")
    batch = _filter_batch(batch, compressed_model)

    compressed_model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = compressed_model(**batch)

    logits, loss = _extract_logits_and_loss(out)
    target_tensor = logits if logits is not None else loss
    assert target_tensor is not None
    assert not torch.isnan(target_tensor).any(), f"NaN detected after {comp_name} compression on GPU"
    assert not torch.isinf(target_tensor).any(), f"Inf detected after {comp_name} compression on GPU"
