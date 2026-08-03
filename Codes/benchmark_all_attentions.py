"""
Benchmark script to train and evaluate all attention variants sequentially.
Clears memory after each run and generates an aggregated comparative analysis.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add Codes directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from framework.models import load_model
from framework.attentions import __all__ as available_attentions
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.utils import clear_memory
from framework.trainer.trainer import create_trainer
from framework.trainer.metrics import get_task_metrics
from framework.data.base import DataConfig, TaskType, create_dataloader

def run_benchmark():
    attentions_to_test = ["full", "svd", "xattention", "linformer", "performer", "nystrom"]
    registered = ATTENTION_REGISTRY.keys()
    print(f"Registered attention types: {registered}")
    
    # Filter to registered ones
    valid_attentions = [a for a in attentions_to_test if a.lower() in registered]
    print(f"Running benchmark on: {valid_attentions}")
    
    results = {}
    
    for attn_type in valid_attentions:
        print("\n" + "=" * 70)
        print(f"RUNNING BENCHMARK FOR ATTENTION: {attn_type.upper()}")
        print("=" * 70)
        
        start_t = time.time()
        try:
            # 1. Load model & tokenizer
            model, tokenizer = load_model(
                "bert-base-uncased",
                attention_type=attn_type,
                attention_rank=32,
                task="sequence_classification",
                num_labels=2,
            )
            
            # 2. Load dataset
            train_cfg = DataConfig(
                task_name="sst2",
                task_type=TaskType.SINGLE_CLASSIFICATION,
                text_column="sentence",
                label_column="label",
                max_length=64,
                batch_size=4,
                max_samples=16,
            )
            eval_cfg = DataConfig(
                task_name="sst2",
                task_type=TaskType.SINGLE_CLASSIFICATION,
                text_column="sentence",
                label_column="label",
                max_length=64,
                batch_size=4,
                max_samples=16,
            )
            train_loader = create_dataloader(train_cfg, tokenizer)
            eval_loader = create_dataloader(eval_cfg, tokenizer)
            
            # 3. Create trainer & train
            output_dir = f"./benchmark_runs/{attn_type}"
            trainer = create_trainer(
                model=model,
                train_dataset=train_loader.dataset,
                eval_dataset=eval_loader.dataset,
                output_dir=output_dir,
                epochs=1,
                batch_size=4,
                compute_metrics=get_task_metrics("sst2"),
                logging_steps=1,
            )
            
            train_res = trainer.train()
            eval_metrics = trainer.evaluate()
            elapsed = time.time() - start_t
            
            metrics_summary = {
                "status": "SUCCESS",
                "train_loss": getattr(train_res, "training_loss", None),
                "eval_loss": eval_metrics.get("eval_loss"),
                "eval_accuracy": eval_metrics.get("eval_accuracy"),
                "elapsed_seconds": round(elapsed, 2),
            }
            results[attn_type] = metrics_summary
            print(f"✅ {attn_type.upper()} Completed: {metrics_summary}")
            
        except Exception as e:
            print(f"❌ {attn_type.upper()} Failed: {e}")
            results[attn_type] = {"status": "FAILED", "error": str(e)}
        finally:
            clear_memory()
            print(f"GPU/MPS memory cleared for {attn_type}.")

    # Save summary report
    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Table
    md_report = "# Attention Module Benchmark & Analysis Report\n\n"
    md_report += "| Attention Type | Status | Eval Loss | Eval Accuracy | Total Time (s) |\n"
    md_report += "|---|---|---|---|---|\n"
    for k, v in results.items():
        if v["status"] == "SUCCESS":
            md_report += f"| **{k}** | ✅ SUCCESS | {v.get('eval_loss', 0.0):.4f} | {v.get('eval_accuracy', 0.0):.4f} | {v['elapsed_seconds']}s |\n"
        else:
            md_report += f"| **{k}** | ❌ FAILED | N/A | N/A | N/A |\n"

    with open(out_dir / "benchmark_report.md", "w") as f:
        f.write(md_report)

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK COMPARISON")
    print("=" * 70)
    print(md_report)

if __name__ == "__main__":
    run_benchmark()
