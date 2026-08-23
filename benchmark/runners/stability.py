"""Layer 3: Hyperparameter and RNG seed stability auditor."""

from __future__ import annotations
import logging
from typing import Dict, Any, Callable, List
import numpy as np
from datasets import Dataset
from .train import train_model

logger = logging.getLogger("benchmark.stability")


def evaluate_stability(
    model_builder_fn: Callable[..., Any],
    model_name: str,
    config: Dict[str, Any],
    train_dataset: Dataset,
    eval_dataset: Dataset,
    task_name: str = "sst2",
) -> Dict[str, Any]:
    """
    Evaluate model optimization stability across multiple random seeds, learning rates, and batch sizes.

    Args:
        model_builder_fn: Factory function taking model_name and returning (model, status).
        model_name: Identifier of the attention mechanism.
        config: Global benchmark configuration.
        train_dataset: Training split (typically SST-2).
        eval_dataset: Validation split.
        task_name: Task for stability testing.

    Returns:
        Dictionary detailing seed variance, hyperparameter coefficients of variation, and stability verdict.
    """
    seeds: List[int] = config.get("seeds", [42, 123, 777])
    learning_rates: List[float] = config.get("learning_rates", [2e-5, 3e-5, 5e-5])
    batch_sizes: List[int] = [16, 32]

    seed_accuracies: List[float] = []
    lr_accuracies: Dict[str, float] = {}
    batch_accuracies: Dict[str, float] = {}

    failed_seed_count = 0

    # 1. Multi-Seed Stability (Fixed LR=3e-5, BS=32)
    for seed in seeds:
        model, status = model_builder_fn(model_name)
        if model is None or status != "OK":
            failed_seed_count += 1
            seed_accuracies.append(0.0)
            continue

        res = train_model(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            task_name=task_name,
            config=config,
            seed=seed,
            lr=3e-5,
            batch_size=32,
            epochs=2,
        )

        acc = float(res.get("eval_metrics", {}).get("accuracy", res.get("eval_metrics", {}).get("score", 0.0)))
        if res.get("status") != "OK" or acc < 0.55:
            failed_seed_count += 1
        seed_accuracies.append(acc)

    # 2. Multi-LR Robustness (Fixed Seed=42, BS=32)
    for lr in learning_rates:
        model, status = model_builder_fn(model_name)
        if model is None or status != "OK":
            lr_accuracies[str(lr)] = 0.0
            continue

        res = train_model(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            task_name=task_name,
            config=config,
            seed=42,
            lr=lr,
            batch_size=32,
            epochs=2,
        )
        acc = float(res.get("eval_metrics", {}).get("accuracy", res.get("eval_metrics", {}).get("score", 0.0)))
        lr_accuracies[str(lr)] = acc

    # 3. Multi-Batch Size Robustness (Fixed Seed=42, LR=3e-5)
    for bs in batch_sizes:
        model, status = model_builder_fn(model_name)
        if model is None or status != "OK":
            batch_accuracies[str(bs)] = 0.0
            continue

        res = train_model(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            task_name=task_name,
            config=config,
            seed=42,
            lr=3e-5,
            batch_size=bs,
            epochs=2,
        )
        acc = float(res.get("eval_metrics", {}).get("accuracy", res.get("eval_metrics", {}).get("score", 0.0)))
        batch_accuracies[str(bs)] = acc

    # Compute Statistical Metrics
    seed_arr = np.array(seed_accuracies, dtype=np.float32)
    mean_seed_acc = float(np.mean(seed_arr))
    std_seed_acc = float(np.std(seed_arr))

    lr_arr = np.array(list(lr_accuracies.values()), dtype=np.float32)
    mean_lr = float(np.mean(lr_arr))
    std_lr = float(np.std(lr_arr))
    cv_lr = float(std_lr / (mean_lr + 1e-8))

    bs_arr = np.array(list(batch_accuracies.values()), dtype=np.float32)
    mean_bs = float(np.mean(bs_arr))
    std_bs = float(np.std(bs_arr))
    cv_batch = float(std_bs / (mean_bs + 1e-8))

    is_stable = bool(failed_seed_count <= len(seeds) // 2 and mean_seed_acc > 0.60 and cv_lr < 0.25)

    return {
        "seed_accuracies": seed_accuracies,
        "mean_seed_acc": mean_seed_acc,
        "std_seed_acc": std_seed_acc,
        "lr_accuracies": lr_accuracies,
        "batch_accuracies": batch_accuracies,
        "cv_lr": cv_lr,
        "cv_batch": cv_batch,
        "failed_seeds": failed_seed_count,
        "total_seeds": len(seeds),
        "stable": is_stable,
    }
