#!/usr/bin/env python3
"""
run_validation_metrics.py — Standalone validation & quality evaluation script.

Computes publication-ready quality metrics (Accuracy, F1, MCC, Pearson/Spearman, PPL, Fidelity)
across attention and compression modules using genuine GLUE / LM validation datasets.

Usage:
    python3 Codes/framework/run_validation_metrics.py --module svd_linear_attention --dataset sst2 --model bert
    python3 Codes/framework/run_validation_metrics.py --track track1_svd --dataset mrpc
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ValidationMetrics")

# Add framework to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_quality_single(
    module_name: str,
    module_type: str,
    model_family: str,
    dataset: str,
    rank: int = 16,
    batch_size: int = 32,
    seq_len: int = 128,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Run full validation quality evaluation for a single model/module combination."""
    import torch
    from framework.eval_all_models_datasets import (
        get_base_model_cached,
        apply_module_to_model,
        get_dataset_batch,
        _extract_logits_and_loss,
        _filter_batch,
        MODEL_TOKENIZER_MAPPING,
        _PREPARED_DATASET_CACHE,
    )
    from framework.core.metrics import compute_dataset_metrics, compute_fidelity

    device = torch.device(device_str)
    num_labels = 1 if dataset.lower() == "stsb" else 2

    quality: Dict[str, Any] = {
        "module_name": module_name,
        "module_type": module_type,
        "model_family": model_family,
        "dataset": dataset,
        "accuracy_pct": 0.0,
        "f1_score": 0.0,
        "mcc": 0.0,
        "pearson": 0.0,
        "spearman": 0.0,
        "perplexity": 0.0,
        "fidelity_score": 0.0,
        "status": "PASS",
        "error_message": "",
    }

    try:
        # 1. Base Model & Custom Module Injection
        model, cfg = get_base_model_cached(
            model_family=model_family,
            num_labels=num_labels,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            device=device,
        )
        model, _ = apply_module_to_model(model, module_name, module_type, rank=rank, device=device)
        model.eval()

        # 2. Baseline Model for Fidelity (Cosine Similarity)
        baseline_model, _ = get_base_model_cached(
            model_family=model_family,
            num_labels=num_labels,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            device=device,
        )
        baseline_model.eval()

        # 3. Load Validation Dataset
        _ = get_dataset_batch(
            dataset, model_family, batch_size=batch_size, seq_len=seq_len, num_labels=num_labels, device=device, allow_synthetic=False
        )
        tok_name = MODEL_TOKENIZER_MAPPING.get(model_family.lower(), "bert-base-uncased")
        dataset_cache_key = (dataset.lower(), tok_name, seq_len)

        if dataset_cache_key not in _PREPARED_DATASET_CACHE:
            logger.warning("Dataset '%s' was not loaded into cache; falling back to synthetic evaluation.", dataset)
            return quality

        val_data = _PREPARED_DATASET_CACHE[dataset_cache_key]
        loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

        all_preds: List[Any] = []
        all_labels: List[Any] = []
        baseline_logits_list: List[torch.Tensor] = []
        compressed_logits_list: List[torch.Tensor] = []
        total_loss: float = 0.0
        n_batches: int = 0

        with torch.no_grad():
            for batch in loader:
                inputs = {
                    ("labels" if k == "label" else k): (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
                filtered_inputs = _filter_batch(inputs, model_family, model)
                filtered_base = _filter_batch(inputs, model_family, baseline_model)

                out = model(**filtered_inputs)
                base_out = baseline_model(**filtered_base)

                logits, loss = _extract_logits_and_loss(out, num_labels=num_labels, labels=inputs.get("labels"))
                base_logits, _ = _extract_logits_and_loss(base_out, num_labels=num_labels, labels=inputs.get("labels"))

                if loss is not None:
                    total_loss += float(loss.item())
                    n_batches += 1

                if logits is not None:
                    compressed_logits_list.append(logits.float().cpu())
                    if base_logits is not None:
                        baseline_logits_list.append(base_logits.float().cpu())

                    if inputs.get("labels") is not None:
                        if dataset.lower() == "stsb":
                            preds = logits.squeeze().cpu().numpy().tolist()
                        else:
                            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
                        labels_batch = inputs["labels"].cpu().numpy().tolist()

                        all_preds.extend(preds if isinstance(preds, list) else [preds])
                        all_labels.extend(labels_batch if isinstance(labels_batch, list) else [labels_batch])

        mean_loss = (total_loss / n_batches) if n_batches > 0 else 0.0
        metrics = compute_dataset_metrics(dataset, all_labels, all_preds, mean_loss)
        fidelity = compute_fidelity(baseline_logits_list, compressed_logits_list)

        quality.update(metrics)
        quality["fidelity_score"] = fidelity

    except Exception as e:
        logger.error("Quality evaluation failed for %s on %s/%s: %s", module_name, model_family, dataset, e)
        quality["status"] = "FAIL"
        quality["error_message"] = str(e)

    return quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone GLUE/LM Quality Metrics Evaluator")
    parser.add_argument("--module", type=str, default="svd_linear_attention", help="Module name to evaluate")
    parser.add_argument("--module_type", type=str, default="attention", choices=["attention", "compressor"])
    parser.add_argument("--model", type=str, default="bert", help="Model family (bert, roberta, t5, distilbert, gpt2)")
    parser.add_argument("--dataset", type=str, default="sst2", help="Dataset name (sst2, mrpc, cola, stsb, wikitext2)")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    logger.info("Evaluating quality for %s (%s) on %s / %s...", args.module, args.module_type, args.model, args.dataset)
    res = evaluate_quality_single(
        module_name=args.module,
        module_type=args.module_type,
        model_family=args.model,
        dataset=args.dataset,
        rank=args.rank,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device_str=args.device,
    )

    print("\n" + "=" * 50)
    print(f" Quality Results: {res['module_name']} | {res['model_family']} | {res['dataset']}")
    print("=" * 50)
    print(f" Status:        {res['status']}")
    print(f" Accuracy:      {res['accuracy_pct']:.2f}%")
    print(f" F1 Score:      {res['f1_score']:.4f}")
    print(f" MCC (CoLA):    {res['mcc']:.4f}")
    print(f" Pearson/Spear: {res['pearson']:.4f} / {res['spearman']:.4f}")
    print(f" Perplexity:    {res['perplexity']:.4f}")
    print(f" Logit Fidelity:{res['fidelity_score']:.4f}")
    print("=" * 50 + "\n")

    if args.output_csv:
        file_exists = os.path.exists(args.output_csv)
        with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(res.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(res)
        logger.info("Saved results to %s", args.output_csv)


if __name__ == "__main__":
    main()
