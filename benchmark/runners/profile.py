"""Layer 2: Computational efficiency, latency, throughput, and VRAM memory scaling profiler."""

from __future__ import annotations
import time
from typing import Dict, Any, List
import torch
import torch.nn as nn
from ..utils.memory import measure_gpu_memory
from ..utils.flops import compute_flops


def synchronize_device(device: torch.device) -> None:
    """Ensure all asynchronous device operations have completed."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def profile_model(
    model: nn.Module,
    seq_lengths: List[int] = [128, 256, 512],
    batch_sizes: List[int] = [1, 32],
    num_warmup: int = 5,
    num_iterations: int = 50,
) -> Dict[str, Any]:
    """
    Profile inference execution throughput, latency, FLOPs, and peak memory across sequence horizons.

    Args:
        model: Model instance to evaluate.
        seq_lengths: List of token sequence lengths.
        batch_sizes: List of batch sizes.
        num_warmup: Number of initial untimed warmup iterations.
        num_iterations: Number of timed forward iterations.

    Returns:
        Structured profile metrics dictionary per (batch_size, sequence_length) configuration.
    """
    model.eval()
    device = next(model.parameters()).device

    profile_results: Dict[str, Any] = {}
    scaling_curve: Dict[str, List[Dict[str, Any]]] = {"by_seq_len": []}

    for b in batch_sizes:
        for l in seq_lengths:
            key = f"bs{b}_seq{l}"
            dummy_input_ids = torch.randint(0, 1000, (b, l), device=device)
            dummy_mask = torch.ones((b, l), device=device)
            inputs = {"input_ids": dummy_input_ids, "attention_mask": dummy_mask}

            # Estimate theoretical FLOPs
            gflops = compute_flops(model, inputs)

            # Warmup
            try:
                with torch.no_grad():
                    for _ in range(num_warmup):
                        model(**inputs)
                    synchronize_device(device)

                # Reset peak memory
                _ = measure_gpu_memory()

                # Timed iterations
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(num_iterations):
                        model(**inputs)
                    synchronize_device(device)
                elapsed = time.time() - start_time

                peak_vram = measure_gpu_memory()
                total_tokens = b * l * num_iterations
                tokens_per_sec = total_tokens / max(elapsed, 1e-6)
                latency_ms = (elapsed / num_iterations) * 1000.0

                entry = {
                    "batch_size": b,
                    "seq_len": l,
                    "tokens_per_sec": float(tokens_per_sec),
                    "latency_ms": float(latency_ms),
                    "peak_vram_mb": float(peak_vram),
                    "gflops": float(gflops) if gflops is not None else 0.0,
                    "status": "OK",
                }
                profile_results[key] = entry
                scaling_curve["by_seq_len"].append(entry)

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                err_msg = str(e).lower()
                if "out of memory" in err_msg or "oom" in err_msg:
                    profile_results[key] = {
                        "batch_size": b,
                        "seq_len": l,
                        "tokens_per_sec": None,
                        "latency_ms": None,
                        "peak_vram_mb": "OOM",
                        "gflops": gflops,
                        "status": "OOM",
                    }
                else:
                    profile_results[key] = {
                        "batch_size": b,
                        "seq_len": l,
                        "tokens_per_sec": None,
                        "latency_ms": None,
                        "peak_vram_mb": "FAIL",
                        "error": str(e),
                        "status": "FAIL",
                    }

    return {
        "configurations": profile_results,
        "scaling_curve": scaling_curve,
    }
