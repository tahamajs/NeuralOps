#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Benchmark Command

Implements the command-line interface for running attention and kernel benchmarks.
"""

import argparse
import sys
import subprocess
import torch


def cmd_bench(args):
    target = getattr(args, "module", None) or getattr(args, "target", None)
    mode = getattr(args, "mode", "eval")
    model = getattr(args, "model", "gpt2")
    dataset = getattr(args, "dataset", "sst2")
    rank = getattr(args, "rank", 32)
    device = getattr(args, "device", "cuda")

    print("==================================================")
    print("NeuralOps Benchmark Runner")
    print(f"Target:   {target}")
    print(f"Mode:     {mode}")
    print(f"Model:    {model} | Dataset: {dataset} | Rank: {rank} | Device: {device}")
    print("==================================================")

    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    import neuralops.attentions
    import neuralops.compression_engine

    all_attn = getattr(args, "all_attention", False) or getattr(args, "all", False)
    all_comp = getattr(args, "all_compressors", False) or getattr(args, "all", False)

    targets = []
    if all_attn:
        targets.extend([(k, "attention") for k in sorted(ATTENTION_REGISTRY.keys())])
    if all_comp:
        targets.extend([(k, "compressor") for k in sorted(COMPRESSOR_REGISTRY.keys())])

    if not targets:
        if not target:
            print("Error: --module or sweep flag (--all-attention, --all-compressors, --all) is required.", file=sys.stderr)
            return 1
        mod_type = "attention" if target in ATTENTION_REGISTRY else "compressor"
        targets = [(target, mod_type)]

    output_dir = getattr(args, "output_dir", None)
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

    if len(targets) > 1:
        print("==================================================")
        print(f"NeuralOps Batch Sweep Runner: {len(targets)} modules queued")
        print(f"Mode: {mode} | Model: {model} | Dataset: {dataset} | Device: {device}")
        print("==================================================")

        import json
        import time
        results = []
        passes = 0
        failures = 0

        for idx, (mod_name, mod_type) in enumerate(targets, 1):
            print(f"\n[{idx}/{len(targets)}] Evaluating {mod_type}: {mod_name}...")
            start_t = time.time()
            try:
                if mode == "kd":
                    cmd = [
                        sys.executable,
                        "neuralops/benchmarks/_legacy/kd_quick_test.py",
                        "--modules", mod_name,
                        "--type", mod_type,
                        "--model_name", model,
                        "--rank", str(rank),
                        "--methods", "C",
                        "--use_cache",
                        "--batch_size", str(getattr(args, "batch_size", 4)),
                        "--grad_accum_steps", str(getattr(args, "grad_accum_steps", 2)),
                        "--recovery_steps", str(getattr(args, "recovery_steps", 2)),
                        "--self_train_steps", str(getattr(args, "self_train_steps", 2)),
                        "--learning_rate", str(getattr(args, "learning_rate", 3e-4)),
                        "--device", device,
                    ]
                    code = subprocess.call(cmd)
                    status = "PASS" if code == 0 else "FAIL"
                    err = None if code == 0 else f"Process exited with code {code}"
                    entry = {"module": mod_name, "type": mod_type, "status": status, "error": err, "elapsed_s": round(time.time() - start_t, 2)}
                else:
                    try:
                        from evaluation.eval_pipeline import evaluate_single_combination_opt
                    except ImportError:
                        from neuralops.evaluation.eval_pipeline import evaluate_single_combination_opt
                    dev = torch.device(device)
                    res = evaluate_single_combination_opt(
                        track="cli_bench",
                        module_name=mod_name,
                        module_type=mod_type,
                        model_family=model,
                        dataset=dataset,
                        rank=rank,
                        device=dev,
                    )
                    d = res.to_dict() if hasattr(res, "to_dict") else dict(res)
                    status = "PASS" if d.get("status") in ("PASS", "PASS_NO_COMPRESSION") else "FAIL"
                    entry = {"module": mod_name, "type": mod_type, "status": status, "metrics": d, "elapsed_s": round(time.time() - start_t, 2)}

                if status == "PASS":
                    passes += 1
                    print(f"  --> PASS ({entry['elapsed_s']}s)")
                else:
                    failures += 1
                    print(f"  --> FAIL: {entry.get('error') or entry.get('metrics', {}).get('error_message')}")

            except Exception as e:
                failures += 1
                entry = {"module": mod_name, "type": mod_type, "status": "FAIL", "error": str(e), "elapsed_s": round(time.time() - start_t, 2)}
                print(f"  --> FAIL (Exception): {e}")

            results.append(entry)

            if output_dir:
                with open(os.path.join(output_dir, "sweep_progress.json"), "w") as f:
                    json.dump(results, f, indent=2)

        summary_path = os.path.join(output_dir, "sweep_summary.json") if output_dir else None
        if summary_path:
            with open(summary_path, "w") as f:
                json.dump({"total": len(targets), "passed": passes, "failed": failures, "results": results}, f, indent=2)
            print(f"\nSaved full sweep results to {summary_path}")

        print(f"\nSweep Finished: {passes}/{len(targets)} passed, {failures} failed.")
        return 0 if failures == 0 else 1

    mod_name, mod_type = targets[0]
    if mode == "kd":
        cmd = [
            sys.executable,
            "neuralops/benchmarks/_legacy/kd_quick_test.py",
            "--modules", mod_name,
            "--type", mod_type,
            "--model_name", model,
            "--rank", str(rank),
            "--methods", "C",
            "--use_cache",
            "--batch_size", str(getattr(args, "batch_size", 4)),
            "--grad_accum_steps", str(getattr(args, "grad_accum_steps", 2)),
            "--recovery_steps", str(getattr(args, "recovery_steps", 50)),
            "--self_train_steps", str(getattr(args, "self_train_steps", 20)),
            "--learning_rate", str(getattr(args, "learning_rate", 3e-4)),
            "--device", device,
        ]
        return subprocess.call(cmd)
    else:
        try:
            from evaluation.eval_pipeline import evaluate_single_combination_opt
        except ImportError:
            from neuralops.evaluation.eval_pipeline import evaluate_single_combination_opt
        dev = torch.device(device)
        res = evaluate_single_combination_opt(
            track="cli_bench",
            module_name=mod_name,
            module_type=mod_type,
            model_family=model,
            dataset=dataset,
            rank=rank,
            device=dev,
        )
        print("\nBenchmark completed successfully:")
        print(res.to_dict() if hasattr(res, "to_dict") else res)
        return 0
