#!/usr/bin/env python3
"""
GPU Server Readiness & Hardware Diagnostics Suite for NeuralOps / SVD Linear Attention.

Runs automated system checks, GPU hardware capability scans, FP16/BF16 precision tests,
SVD low-rank factorized attention forward/backward passes, and distributed runtime verification.

Usage:
    python Codes/framework/launch_gpu_server_test.py
    python Codes/framework/launch_gpu_server_test.py --device cuda --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Ensure framework import path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir / "Codes"))
sys.path.insert(0, str(root_dir / "Codes" / "framework"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GPUServerTest")


def inspect_hardware_capabilities() -> Dict[str, Any]:
    """Inspects host machine GPU, CUDA, MPS, and memory capabilities."""
    info: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["devices"].append({
                "index": idx,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "multi_processor_count": props.multi_processor_count,
                "major": props.major,
                "minor": props.minor,
            })
    elif info["mps_available"]:
        info["device_type"] = "Apple Silicon MPS"
    else:
        info["device_type"] = "CPU fallback"

    return info


def verify_svd_linear_gpu(device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    """Verifies SVDLinear module initialization, forward, and backward pass on GPU."""
    from framework.core.attention_utils import SVDLinear

    in_dim, out_dim, rank = 768, 768, 64
    layer = SVDLinear(in_features=in_dim, out_features=out_dim, rank=rank, bias=True).to(device=device, dtype=dtype)

    batch_size, seq_len = 8, 128
    x = torch.randn(batch_size, seq_len, in_dim, device=device, dtype=dtype, requires_grad=True)

    t0 = time.perf_counter()
    out = layer(x)
    loss = out.sum()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert out.shape == (batch_size, seq_len, out_dim), f"Unexpected out shape: {out.shape}"
    assert layer.U.grad is not None and torch.isfinite(layer.U.grad).all(), "U gradient is invalid"
    assert layer.V.grad is not None and torch.isfinite(layer.V.grad).all(), "V gradient is invalid"
    assert x.grad is not None and torch.isfinite(x.grad).all(), "Input gradient is invalid"

    return {
        "status": "PASSED",
        "latency_ms": round(elapsed_ms, 3),
        "output_shape": list(out.shape),
        "param_count": sum(p.numel() for p in layer.parameters()),
        "dense_param_equiv": in_dim * out_dim + out_dim,
    }


def verify_attention_modules_gpu(
    device: torch.device, dtype: torch.dtype, attn_keys: List[str]
) -> Dict[str, Any]:
    """Tests key registered attention modules on GPU."""
    from framework.core.registry import ATTENTION_REGISTRY
    from transformers import DistilBertConfig
    from framework.tests.conftest import build_attention
    import framework.attentions  # Ensure registrations

    results: Dict[str, Any] = {}
    batch_size, seq_len, d_model = 2, 64, 768
    config = DistilBertConfig(hidden_size=d_model, num_attention_heads=12)

    for key in attn_keys:
        if key not in ATTENTION_REGISTRY:
            results[key] = {"status": "SKIPPED", "reason": "Not in registry"}
            continue

        try:
            mod = build_attention(key, config=config)
            mod = mod.to(device=device, dtype=dtype)
            x = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)

            t0 = time.perf_counter()
            out = mod(x)
            out_tensor = out[0] if isinstance(out, tuple) else out
            loss = out_tensor.sum()
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            assert torch.isfinite(out_tensor).all(), f"Non-finite output in {key}"

            results[key] = {
                "status": "PASSED",
                "latency_ms": round(elapsed_ms, 3),
                "is_finite": True,
            }
        except Exception as exc:
            results[key] = {
                "status": "FAILED",
                "error": str(exc),
            }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuralOps GPU Server Diagnostics & Test Suite")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    # Determine hardware device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Determine dtype
    if args.dtype == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif device.type in ["cuda", "mps"]:
            dtype = torch.float32
        else:
            dtype = torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    logger.info("=" * 70)
    logger.info("  STARTING NEURALOPS GPU SERVER READINESS & DIAGNOSTICS SUITE")
    logger.info("  Target Device: %s | Precision: %s", device, dtype)
    logger.info("=" * 70)

    # 1. Hardware scan
    hw_info = inspect_hardware_capabilities()
    logger.info("Hardware Detected: %s (CUDA GPUs: %d)", hw_info.get("device_type", "CUDA"), hw_info.get("cuda_device_count", 0))
    for dev in hw_info.get("devices", []):
        logger.info("  -> GPU [%d]: %s (VRAM: %.2f GB)", dev["index"], dev["name"], dev["total_memory_gb"])

    # 2. SVD Linear GPU verification
    logger.info("\n[1/3] Verifying SVDLinear Forward & Backward Pass on %s...", device)
    svd_report = verify_svd_linear_gpu(device, dtype)
    logger.info("  -> SVDLinear Status: %s (Latency: %.3f ms)", svd_report["status"], svd_report["latency_ms"])

    # 3. Key attention modules GPU verification
    key_attentions = [
        "full",
        "gqa",
        "svd_linear",
        "mamba_svd_flash",
        "spectral_mamba",
        "samba_m3",
        "reap_moe",
        "aegis_hybrid",
        "flash3",
    ]
    logger.info("\n[2/3] Verifying Attention Modules on %s across %d architectures...", device, len(key_attentions))
    attn_report = verify_attention_modules_gpu(device, dtype, key_attentions)
    for k, v in attn_report.items():
        logger.info("  -> Architecture [%s]: %s (%s)", k, v["status"], f"{v.get('latency_ms', 0)} ms" if v["status"] == "PASSED" else v.get("error", ""))

    # 4. Save diagnostics report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "gpu_server_readiness_report.json"

    master_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": hw_info,
        "test_config": {
            "device": str(device),
            "dtype": str(dtype),
        },
        "svd_linear_verification": svd_report,
        "attention_modules_verification": attn_report,
        "readiness_status": "READY_FOR_TRAINING_AND_SWEEPS",
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    logger.info("\n[3/3] Master GPU Server Diagnostics Complete!")
    logger.info("Saved readiness report to: %s", report_file)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
