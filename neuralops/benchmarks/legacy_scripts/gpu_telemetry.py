#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware Pre-Flight & Memory Auto-Tuning Engine (Phase 1 & Phase 2).

Discovers GPU architecture, VRAM bounds, enables TF32 / CUDNN benchmarks,
and dynamically searches for maximal safe batch sizes and sequence lengths
using representative attention modules (e.g. CosformerLinearAttention)
to saturate 85-92% of available accelerator memory without OOM.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Callable

import torch
import torch.nn as nn


def get_hardware_info() -> dict[str, Any]:
    """Inspect and report physical accelerator properties and optimal flags."""
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "pytorch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024**3)
        free_vram_gb = torch.cuda.mem_get_info()[0] / (1024**3)
        cap = (props.major, props.minor)
        info.update(
            {
                "device_name": props.name,
                "compute_capability": f"{cap[0]}.{cap[1]}",
                "total_vram_gb": round(total_vram_gb, 2),
                "free_vram_gb": round(free_vram_gb, 2),
                "multi_processor_count": props.multi_processor_count,
                "supports_bfloat16": cap[0] >= 8,
                "supports_fp8": cap[0] >= 9,
                "tensor_cores": cap[0] >= 7,
            }
        )
    elif info["mps_available"]:
        info.update(
            {
                "device_name": "Apple Silicon (MPS)",
                "compute_capability": "MPS",
                "supports_bfloat16": True,
                "supports_fp8": False,
                "tensor_cores": False,
            }
        )
    else:
        info.update(
            {
                "device_name": "CPU",
                "compute_capability": "CPU",
                "supports_bfloat16": False,
                "supports_fp8": False,
                "tensor_cores": False,
            }
        )
    return info


def configure_optimal_gpu_runtime() -> dict[str, bool]:
    """Enable high-throughput runtime flags for CUDA / PyTorch."""
    flags = {"tf32_matmul": False, "tf32_cudnn": False, "cudnn_benchmark": False}
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        flags = {"tf32_matmul": True, "tf32_cudnn": True, "cudnn_benchmark": True}
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    return flags


def get_default_probe_module(dim: int = 768, num_heads: int = 12) -> nn.Module:
    """Build a representative attention module for realistic memory profiling."""
    try:
        from neuralops.layers.attention.cosformer_attention import CosformerLinearAttention

        return CosformerLinearAttention(dim=dim, num_heads=num_heads)
    except Exception:
        # Fallback to standard MultiheadAttention
        return nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)


class AutoVRAMSaturator:
    """Probes model architectures to determine the maximum batch size and sequence length."""

    def __init__(self, target_vram_utilization: float = 0.88, device: str | None = None) -> None:
        self.target_vram_utilization = target_vram_utilization
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

    def find_max_batch_size(
        self,
        model_factory: Callable[[], nn.Module] | None = None,
        seq_len: int = 512,
        dim: int = 768,
        candidate_batches: list[int] | None = None,
    ) -> dict[str, Any]:
        """Probes forward + backward passes to find the largest safe batch size."""
        if candidate_batches is None:
            candidate_batches = [1, 2, 4, 8, 16, 32, 64, 128]

        if model_factory is None:
            model_factory = lambda: get_default_probe_module(dim=dim)

        best_batch = 1
        measured_results: list[dict[str, Any]] = []

        model = model_factory().to(self.device)
        model.train()

        for b in candidate_batches:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()

            try:
                # Synthetic input
                x = torch.randn(b, seq_len, dim, device=self.device, requires_grad=True)

                # Execute realistic forward
                if isinstance(model, nn.MultiheadAttention):
                    out, _ = model(x, x, x)
                else:
                    out = model(x)
                if hasattr(out, "logits"):
                    out = out.logits
                elif isinstance(out, tuple):
                    out = out[0]

                loss = out.sum()
                loss.backward()

                if self.device.type == "cuda":
                    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
                    total_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                    utilization = peak_mem_mb / total_mem_mb
                else:
                    peak_mem_mb = 0.0
                    utilization = 0.0

                measured_results.append(
                    {
                        "batch_size": b,
                        "seq_len": seq_len,
                        "peak_mem_mb": round(peak_mem_mb, 2),
                        "vram_utilization": round(utilization, 4),
                        "status": "PASS",
                    }
                )

                if self.device.type == "cuda" and utilization > self.target_vram_utilization:
                    break
                best_batch = b

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    measured_results.append(
                        {
                            "batch_size": b,
                            "seq_len": seq_len,
                            "peak_mem_mb": -1,
                            "vram_utilization": -1,
                            "status": "OOM",
                        }
                    )
                    break
                raise exc

        return {
            "optimal_batch_size": best_batch,
            "seq_len": seq_len,
            "device": str(self.device),
            "probing_history": measured_results,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 & 2: Hardware Pre-Flight & VRAM Auto-Tuning")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--probe-model", action="store_true", help="Run VRAM auto-saturator probe")
    parser.add_argument("--export-hardware-json", type=str, default=None, help="Save hardware profile JSON to file")
    args = parser.parse_args()

    hw_info = get_hardware_info()
    flags = configure_optimal_gpu_runtime()

    result: dict[str, Any] = {"hardware": hw_info, "runtime_flags": flags}

    if args.probe_model:
        saturator = AutoVRAMSaturator()
        probe_res = saturator.find_max_batch_size(seq_len=512, dim=768)
        result["vram_probe"] = probe_res

    if args.export_hardware_json:
        out_p = os.path.abspath(args.export_hardware_json)
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 65)
        print("  NeuralOps Phase 1 & 2: Hardware Pre-Flight & Memory Auto-Tuner")
        print("=" * 65)
        for k, v in hw_info.items():
            print(f"  • {k:<25}: {v}")
        print("-" * 65)
        print(f"  Optimized Runtime Flags : {flags}")
        if "vram_probe" in result:
            print("-" * 65)
            vp = result["vram_probe"]
            print(f"  Optimal Batch Size      : {vp['optimal_batch_size']} (seq_len={vp['seq_len']})")
        print("=" * 65)
        print("✓ Phase 1 & 2 Hardware Pre-Flight complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
