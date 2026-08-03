"""
Comprehensive multi-task, multi-model, and multi-attention benchmark runner.
Runs full multi-epoch experiments across models, datasets, and attention modules.
Supports checkpoint resuming, clears memory after each run, and generates aggregated reports.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from framework.models import load_model
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.utils import clear_memory
from framework.trainer.trainer import create_trainer
from framework.trainer.metrics import get_task_metrics
from framework.data.base import DataConfig, TaskType, create_dataloader

from framework.core.memory import cleanup_gpu_memory, get_gpu_memory_stats

DEFAULT_TASKS = ["sst2", "mrpc", "cola"]
DEFAULT_MODELS = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base"]
DEFAULT_ATTENTIONS = [
    "full",
    "svd",
    "xattention",
    "linformer",
    "performer",
    "nystrom",
    "asvd",
    "gqa",
    "lsh",
]


def resolve_task_type(task_name: str) -> TaskType:
    if task_name in ["mrpc", "stsb", "rte", "qqp", "mnli", "qnli"]:
        return TaskType.PAIR_CLASSIFICATION
    return TaskType.SINGLE_CLASSIFICATION


def run_benchmark(
    tasks=None,
    models=None,
    attentions=None,
    epochs=3,
    batch_size=16,
    max_samples=1000,
    resume=True,
):
    registered_keys = list(ATTENTION_REGISTRY.keys())
    print(f"Registered attention types in framework: {len(registered_keys)} modules")
    
    if tasks is None or "all" in tasks:
        tasks = DEFAULT_TASKS
    if models is None or "all" in models:
        models = DEFAULT_MODELS
    if attentions is None or "all" in attentions:
        attentions = [a for a in DEFAULT_ATTENTIONS if a in registered_keys]

    print(f"Tasks: {tasks}")
    print(f"Models: {models}")
    print(f"Attentions: {attentions}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Max Samples: {max_samples}")

    results = {}
    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)
    summary_file = out_dir / "full_summary.json"

    if summary_file.exists() and resume:
        try:
            with open(summary_file) as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {summary_file}")
        except Exception as e:
            print(f"Could not load previous summary: {e}")

    for task in tasks:
        task_type = resolve_task_type(task)
        for model_name in models:
            for attn_type in attentions:
                run_key = f"{task}___{model_name}___{attn_type}"
                
                if resume and run_key in results and results[run_key].get("status") == "SUCCESS":
                    print(f"Skipping already completed run: {run_key}")
                    continue

                print("\n" + "=" * 80)
                print(f"RUNNING EXPERIMENT: Task={task.upper()} | Model={model_name} | Attention={attn_type.upper()}")
                print("=" * 80)

                cleanup_gpu_memory(force=True)
                start_t = time.time()
                try:
                    # 1. Load model & tokenizer
                    model, tokenizer = load_model(
                        model_name,
                        attention_type=attn_type,
                        attention_rank=32,
                        task="sequence_classification",
                        num_labels=2 if task != "mnli" else 3,
                    )

                    # 2. Load dataset
                    train_cfg = DataConfig(
                        task_name=task,
                        task_type=task_type,
                        max_length=64,
                        batch_size=batch_size,
                        max_samples=max_samples,
                    )
                    eval_cfg = DataConfig(
                        task_name=task,
                        task_type=task_type,
                        max_length=64,
                        batch_size=batch_size,
                        max_samples=max_samples,
                    )
                    train_loader = create_dataloader(train_cfg, tokenizer)
                    eval_loader = create_dataloader(eval_cfg, tokenizer)

                    # 3. Create trainer & execute full training
                    out_name = f"{task}_{model_name.replace('/', '_')}_{attn_type}"
                    output_dir = f"./benchmark_runs/{out_name}"
                    trainer = create_trainer(
                        model=model,
                        train_dataset=train_loader.dataset,
                        eval_dataset=eval_loader.dataset,
                        output_dir=output_dir,
                        epochs=epochs,
                        batch_size=batch_size,
                        compute_metrics=get_task_metrics(task),
                        logging_steps=10,
                    )

                    train_res = trainer.train()
                    eval_metrics = trainer.evaluate()
                    elapsed = time.time() - start_t

                    metrics_summary = {
                        "task": task,
                        "model": model_name,
                        "attention": attn_type,
                        "epochs": epochs,
                        "status": "SUCCESS",
                        "train_loss": getattr(train_res, "training_loss", None),
                        "eval_loss": eval_metrics.get("eval_loss"),
                        "eval_accuracy": eval_metrics.get("eval_accuracy") or eval_metrics.get("accuracy"),
                        "elapsed_seconds": round(elapsed, 2),
                    }
                    results[run_key] = metrics_summary
                    print(f"✅ SUCCESS: {run_key} -> {metrics_summary}")

                    # Incremental JSON save
                    with open(summary_file, "w") as f:
                        json.dump(results, f, indent=2)

                except Exception as e:
                    print(f"❌ FAILED: {run_key} -> {e}")
                    results[run_key] = {
                        "task": task,
                        "model": model_name,
                        "attention": attn_type,
                        "status": "FAILED",
                        "error": str(e),
                    }
                    with open(summary_file, "w") as f:
                        json.dump(results, f, indent=2)
                finally:
                    # Clean up references and flush memory
                    del model, tokenizer, train_loader, eval_loader
                    cleanup_gpu_memory(force=True)
                    print(f"GPU/MPS memory cleared for {run_key}.")

    # Generate Markdown Table
    md_report = "# Full Multi-Task, Multi-Model & Multi-Attention Benchmark Report\n\n"
    md_report += "| Task | Model | Attention Type | Epochs | Status | Eval Loss | Eval Accuracy | Total Time (s) |\n"
    md_report += "|---|---|---|---|---|---|---|---|\n"
    for k, v in results.items():
        if v["status"] == "SUCCESS":
            e_loss = f"{v.get('eval_loss'):.4f}" if v.get('eval_loss') is not None else "N/A"
            e_acc = f"{v.get('eval_accuracy'):.4f}" if v.get('eval_accuracy') is not None else "N/A"
            md_report += f"| **{v['task']}** | **{v['model']}** | **{v['attention']}** | {v.get('epochs', 1)} | ✅ SUCCESS | {e_loss} | {e_acc} | {v['elapsed_seconds']}s |\n"
        else:
            md_report += f"| **{v.get('task', 'N/A')}** | **{v.get('model', 'N/A')}** | **{v.get('attention', 'N/A')}** | N/A | ❌ FAILED | N/A | N/A | N/A |\n"

    with open(out_dir / "comprehensive_report.md", "w") as f:
        f.write(md_report)

    # Generate Paper-Ready LaTeX Table
    latex_table = "\\begin{table*}[t]\n\\centering\n\\small\n"
    latex_table += "\\caption{Empirical evaluation across architectures and efficient attention mechanisms.}\n"
    latex_table += "\\label{tab:benchmark_results}\n"
    latex_table += "\\begin{tabular}{lllcccc}\n\\toprule\n"
    latex_table += "\\textbf{Dataset} & \\textbf{Backbone} & \\textbf{Attention Mechanism} & \\textbf{Epochs} & \\textbf{Loss} & \\textbf{Accuracy (\\%)} & \\textbf{Time (s)} \\\\\n\\midrule\n"
    for k, v in results.items():
        if v["status"] == "SUCCESS":
            e_loss = f"{v.get('eval_loss'):.4f}" if v.get('eval_loss') is not None else "--"
            e_acc = f"{v.get('eval_accuracy') * 100:.2f}" if v.get('eval_accuracy') is not None else "--"
            latex_table += f"{v['task'].upper()} & {v['model']} & \\textsc{{{v['attention']}}} & {v.get('epochs', 1)} & {e_loss} & \\textbf{{{e_acc}}} & {v['elapsed_seconds']} \\\\\n"
    latex_table += "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"

    with open(out_dir / "table_latex.tex", "w") as f:
        f.write(latex_table)

    print("\n" + "=" * 80)
    print("FINAL FULL EXPERIMENT BENCHMARK COMPARISON SUMMARY")
    print("=" * 80)
    print(md_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full multi-model & multi-dataset benchmark.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Datasets to benchmark")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to benchmark")
    parser.add_argument("--attentions", nargs="+", default=DEFAULT_ATTENTIONS, help="Attentions to benchmark")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per run")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max train/eval samples")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume previous runs")
    args = parser.parse_args()

    run_benchmark(
        tasks=args.tasks,
        models=args.models,
        attentions=args.attentions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        resume=not args.no_resume,
    )
