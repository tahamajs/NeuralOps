#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Eval ultra long context scaling module.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from neuralops.utils.replace_attention import replace_attention


def _is_cuda_device(device: str | torch.device) -> bool:
    dev_str = str(device).lower()
    return dev_str == "cuda" or dev_str.startswith("cuda:")


def _get_cpu_memory_mb() -> float:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / (1024**2) if sys.platform == "darwin" else ru / 1024.0
    except Exception:
        return 0.0


def print_log(msg: str) -> None:
    """Execute print log routine."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def build_model_for_context(
    model_path: str,
    device: torch.device,
    attention_type: str | None = None,
    max_pos: int = 8192,
) -> torch.nn.Module:
    """Load GPT-2 base and interpolate positional embeddings up to max_pos."""
    base = GPT2LMHeadModel.from_pretrained(model_path)
    config = base.config
    config.max_position_embeddings = max_pos
    config.n_positions = max_pos
    config.n_ctx = max_pos
    model = GPT2LMHeadModel(config)

    model.transformer.wte.load_state_dict(base.transformer.wte.state_dict())
    model.lm_head.load_state_dict(base.lm_head.state_dict())
    model.transformer.ln_f.load_state_dict(base.transformer.ln_f.state_dict())

    orig_wpe = base.transformer.wpe.weight.data
    new_wpe = (
        torch.nn.functional.interpolate(
            orig_wpe.unsqueeze(0).transpose(1, 2), size=max_pos, mode="linear", align_corners=True
        )
        .squeeze(0)
        .transpose(0, 1)
    )
    model.transformer.wpe.weight.data.copy_(new_wpe)

    for i in range(len(base.transformer.h)):
        model.transformer.h[i].load_state_dict(base.transformer.h[i].state_dict())

    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if attention_type is not None:
        replace_attention(model, attention_key=attention_type)

    return model.to(device).eval()


def run_ultra_long_context_scaling(
    device_name: str = "cuda:0",
    seq_lengths: list[int] | None = None,
    model_path: str | None = None,
    output_path: str | Path | None = None,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    """Execute sequence length scaling sweep and record memory/latency metrics."""
    if _is_cuda_device(device_name) and not torch.cuda.is_available():
        if not allow_cpu:
            raise RuntimeError("This ultra-long-context experiment requires CUDA.")
        print_log("WARNING: CUDA requested but unavailable; falling back to CPU via --allow-cpu.")
        device_name = "cpu"

    device = torch.device(device_name)
    if device.type == "cuda":
        print_log(f"Ultra-Long Context Suite on CUDA: {torch.cuda.get_device_name(device)}")
    else:
        print_log("Ultra-Long Context Suite on CPU")

    model_path = model_path or os.environ.get("NEURALOPS_MODEL_PATH", "gpt2")
    _ = AutoTokenizer.from_pretrained(model_path)

    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096, 8192]

    architectures = [
        ("GPT-2 Baseline (MHA)", None),
        ("GPT-2 + SVD Linear Attention", "svd_linear_attention"),
        ("GPT-2 + Gated Delta Net", "gated_delta_net"),
    ]

    long_context_results: dict[str, Any] = {}
    print_log(f"=== Memory Footprint & Latency Scaling (L={min(seq_lengths)}..{max(seq_lengths)}) ===")

    max_pos = max(8192, max(seq_lengths))

    for name, key in architectures:
        print_log(f"\n--- Testing Architecture: {name} ---")
        try:
            model = build_model_for_context(model_path, device, attention_type=key, max_pos=max_pos)
        except Exception as exc:
            print_log(f"Architecture build failed for {name}: {exc}")
            long_context_results[name] = {"build_error": str(exc)}
            continue

        scaling_data: dict[str | int, Any] = {}

        for L in seq_lengths:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            inp = torch.randint(100, 1000, (1, L), device=device)

            try:
                with torch.no_grad():
                    _ = model(inp[:, : min(64, L)])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)

                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(inp)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                peak_vram_mb = (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else _get_cpu_memory_mb()
                )

                nans = torch.isnan(out.logits).sum().item()
                infs = torch.isinf(out.logits).sum().item()

                status = "PASS" if (nans == 0 and infs == 0) else "FAIL_NUMERICS"
                scaling_data[L] = {
                    "latency_ms": round(latency_ms, 2),
                    "peak_vram_mb": round(peak_vram_mb, 2),
                    "nans": nans,
                    "infs": infs,
                    "status": status,
                }
                print_log(
                    f"L={L:5d} | Latency: {latency_ms:7.2f} ms | Peak VRAM: {peak_vram_mb:8.2f} MB | "
                    f"NaNs: {nans} | Infs: {infs} | Status: {status}"
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" in str(exc).lower():
                    print_log(f"L={L:5d} | Out of Memory (OOM)!")
                    scaling_data[L] = {"status": "OOM", "error": str(exc)}
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                print_log(f"L={L:5d} | Error: {str(exc)[:60]}")
                scaling_data[L] = {"status": f"ERR: {str(exc)[:50]}"}
                break
            except Exception as e:
                print_log(f"L={L:5d} | Error: {str(e)[:60]}")
                scaling_data[L] = {"status": f"ERR: {str(e)[:50]}"}
                break
            finally:
                del inp
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        long_context_results[name] = scaling_data

    out_file = output_path or os.path.join(
        os.environ.get("NEURALOPS_OUTPUT_DIR", "."), "ultra_long_context_scaling_results.json"
    )
    out_dir = Path(out_file).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(long_context_results, f, indent=2)

    print_log(f"Long-context scaling sweep finished! Written to {out_file}")
    return long_context_results


def main() -> int:
    """Run this standalone evaluation experiment."""
    parser = argparse.ArgumentParser(description="Ultra-long-context evaluation and memory scaling")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target device (cuda:0, cuda, cpu)")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--model-path", type=str, default=None, help="HuggingFace model checkpoint or directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save JSON results")
    parser.add_argument("--output-file", type=str, default=None, help="Exact output JSON file path")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow fallback to CPU if CUDA is unavailable")
    args = parser.parse_args()

    out_path = args.output_file
    if out_path is None and args.output_dir:
        out_path = os.path.join(args.output_dir, "ultra_long_context_scaling_results.json")

    try:
        run_ultra_long_context_scaling(
            device_name=args.device,
            seq_lengths=args.seq_lengths,
            model_path=args.model_path,
            output_path=out_path,
            allow_cpu=args.allow_cpu,
        )
    except RuntimeError as exc:
        if "requires CUDA" in str(exc) or "CUDA" in str(exc):
            sys.stderr.write(f"Error: {exc}\n")
            return 3  # AGENTS.md §10: 3 backend unavailable
        sys.stderr.write(f"Runtime error: {exc}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"Unexpected error: {exc}\n")
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
