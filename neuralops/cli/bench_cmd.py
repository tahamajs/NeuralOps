"""Robust benchmark command for attention and compressor modules."""
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

    print(f"==================================================")
    print(f"NeuralOps Benchmark Runner")
    print(f"Target:   {target}")
    print(f"Mode:     {mode}")
    print(f"Model:    {model} | Dataset: {dataset} | Rank: {rank} | Device: {device}")
    print(f"==================================================")

    if not target:
        print("Error: --module or --target is required.", file=sys.stderr)
        return 1

    from neuralops.core.registry import ATTENTION_REGISTRY
    mod_type = "attention" if target in ATTENTION_REGISTRY else "compressor"

    if mode == "kd":
        # Safe legacy KD test invocation
        cmd = [
            sys.executable,
            "neuralops/benchmarks/_legacy/kd_quick_test.py",
            "--modules", target,
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
        from neuralops.evaluation.eval_pipeline import evaluate_single_combination_opt
        from neuralops.core.registry import ATTENTION_REGISTRY
        mod_type = "attention" if target in ATTENTION_REGISTRY else "compressor"
        dev = torch.device(device)
        res = evaluate_single_combination_opt(
            track="cli_bench",
            module_name=target,
            module_type=mod_type,
            model_family=model,
            dataset=dataset,
            rank=rank,
            device=dev,
        )
        print("\nBenchmark completed successfully:")
        print(res.to_dict() if hasattr(res, "to_dict") else res)
        return 0
