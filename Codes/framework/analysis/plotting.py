"""Comprehensive plotting functions for model analysis."""
from typing import Any

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import uniform_filter1d

# Default style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)


def set_style() -> None:
    """Set the style."""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    sns.set_context("paper", font_scale=1.2)


# ---------- Individual plot functions ----------


def plot_loss_curve(history, metrics, config, output_dir, opts) -> Any:
    """Plot loss curve."""
    steps = [h.get("step", i) for i, h in enumerate(history) if "loss" in h]
    train = [h["loss"] for h in history if "loss" in h]
    eval_steps = [h.get("step", i) for i, h in enumerate(history) if "eval_loss" in h]
    eval_l = [h["eval_loss"] for h in history if "eval_loss" in h]

    smooth = opts.get("loss_smooth", 0.6)
    if smooth > 0 and len(train) > 5:
        window = max(1, int(len(train) * smooth / 10))
        train = uniform_filter1d(np.array(train), size=window)

    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 6]))
    ax.plot(steps, train, label="Train Loss", color="#3B528B")
    if eval_steps:
        ax.plot(
            eval_steps, eval_l, "o-", label="Eval Loss", color="#21908C", markersize=4
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "loss_curve.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close()
    return [str(path)]


def plot_accuracy_curve(history, metrics, config, output_dir, opts) -> Any:
    """Plot accuracy curve."""
    steps = [h.get("step", i) for i, h in enumerate(history) if "eval_accuracy" in h]
    acc = [h["eval_accuracy"] for h in history if "eval_accuracy" in h]
    if not steps:
        return []
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 6]))
    ax.plot(steps, acc, "o-", color="#5DC963", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "accuracy_curve.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close()
    return [str(path)]


def plot_lr_schedule(history, metrics, config, output_dir, opts) -> Any:
    """Plot lr schedule."""
    steps = [h.get("step", i) for i, h in enumerate(history) if "learning_rate" in h]
    lr = [h["learning_rate"] for h in history if "learning_rate" in h]
    if not steps:
        return []
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 4]))
    ax.plot(steps, lr, color="#FDE725")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "lr_schedule.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close()
    return [str(path)]


def plot_training_time(history, metrics, config, output_dir, opts) -> Any:
    """Plot training time."""
    times = [
        h.get("elapsed", i * 0.01)
        for i, h in enumerate(history)
        if "elapsed" in h or "step" in h
    ]
    if not times:
        return []
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 4]))
    ax.plot(range(len(times)), times, color="#440154")
    ax.set_xlabel("Step")
    ax.set_ylabel("Time (s)")
    ax.set_title("Training Time vs Steps")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "training_time.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close()
    return [str(path)]


def plot_confusion_matrix(history, metrics, config, output_dir, opts) -> Any:
    """Plot confusion matrix."""
    preds_path = Path(output_dir).parent / "artifacts" / "eval_predictions.json"
    if not preds_path.exists():
        return []
    with open(preds_path) as f:
        data = json.load(f)
    labels = data.get("labels", [])
    preds = data.get("predictions", [])
    if not labels:
        return []
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds)
    class_names = opts.get("class_names", None)
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 8]))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    path = Path(output_dir) / "confusion_matrix.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close()
    return [str(path)]


def plot_gradient_flow(history, metrics, config, output_dir, opts) -> Any:
    """Plot gradient flow."""
    steps = [h.get("step", i) for i, h in enumerate(history) if "grad_norm" in h]
    norms = [h["grad_norm"] for h in history if "grad_norm" in h]
    if not steps:
        return []
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 4]))
    ax.plot(steps, norms, color="#440154")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Flow")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "gradient_flow.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close()
    return [str(path)]


def plot_gpu_memory(history, metrics, config, output_dir, opts) -> Any:
    """Plot gpu memory."""
    gpu_steps = [h.get("step", i) for i, h in enumerate(history) if "gpu_mem" in h]
    gpu = [h["gpu_mem"] for h in history if "gpu_mem" in h]
    cpu_steps = [h.get("step", i) for i, h in enumerate(history) if "cpu_mem" in h]
    cpu = [h["cpu_mem"] for h in history if "cpu_mem" in h]
    if not gpu_steps and not cpu_steps:
        return []
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 4]))
    if gpu_steps:
        ax.plot(gpu_steps, gpu, label="GPU Mem (MB)", color="#3B528B")
    if cpu_steps:
        ax.plot(cpu_steps, cpu, label="CPU Mem (MB)", color="#5DC963")
    ax.set_xlabel("Step")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory Usage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "gpu_memory.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close()
    return [str(path)]


def plot_parameter_distribution(history, metrics, config, output_dir, opts) -> Any:
    """Plot parameter distribution."""
    param_path = Path(output_dir).parent / "artifacts" / "param_counts.json"
    if not param_path.exists():
        return []
    with open(param_path) as f:
        param_data = json.load(f)
    layers = list(param_data.keys())
    counts = list(param_data.values())
    fig, ax = plt.subplots(figsize=opts.get("figsize", [8, 6]))
    ax.barh(layers, counts, color="#21908C")
    ax.set_xlabel("Parameter Count")
    ax.set_title("Parameter Distribution per Layer")
    plt.tight_layout()
    path = Path(output_dir) / "parameter_count.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close()
    return [str(path)]


def plot_rank_allocation(history, metrics, config, output_dir, opts) -> Any:
    """Plot rank allocation."""
    rank_path = Path(output_dir).parent / "artifacts" / "rank_allocation.json"
    if not rank_path.exists():
        return []
    with open(rank_path) as f:
        rank_data = json.load(f)
    layers = list(rank_data.keys())
    ranks = list(rank_data.values())
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 4]))
    ax.bar(layers, ranks, color="#5DC963")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Selected Rank")
    ax.set_title("Rank Allocation per Layer")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "rank_allocation.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close()
    return [str(path)]


def plot_radar_chart(history, metrics, config, output_dir, opts) -> Any:
    """Plot radar chart."""
    if not metrics:
        return []
    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    if len(numeric) < 3:
        return []
    categories = list(numeric.keys())
    values = list(numeric.values())
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values = values + values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(
        figsize=opts.get("figsize", [8, 8]), subplot_kw=dict(polar=True)
    )
    ax.fill(angles, values, alpha=0.25, color="#3B528B")
    ax.plot(angles, values, color="#3B528B", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Performance Radar Chart")
    plt.tight_layout()
    path = Path(output_dir) / "radar_chart.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close()
    return [str(path)]


def plot_attention_heatmap(history, metrics, config, output_dir, opts) -> Any:
    """Plot attention heatmap."""
    attn_path = Path(output_dir).parent / "artifacts" / "attention_sample.json"
    if not attn_path.exists():
        return []
    with open(attn_path) as f:
        attn_data = json.load(f)
    matrix = np.array(attn_data.get("matrix", []))
    tokens = attn_data.get("tokens", None)
    layer = attn_data.get("layer", 0)
    head = attn_data.get("head", 0)
    # Simple heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, cmap="viridis", xticklabels=tokens, yticklabels=tokens, ax=ax)
    ax.set_title(f"Attention Heatmap – Layer {layer} Head {head}")
    path = Path(output_dir) / "attention_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return [str(path)]


# ---------- Comprehensive plots ----------


def create_comprehensive_plots(
    df: pd.DataFrame, output_dir: str, figsize: tuple[int, int] = (18, 16)
) -> None:
    """Build and return comprehensive plots."""
    os.makedirs(output_dir, exist_ok=True)
    set_style()
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle(
        "SVD Compression Analysis: Comprehensive Results",
        fontsize=16,
        fontweight="bold",
    )
    # 1. Accuracy vs Rank by model
    ax = axes[0, 0]
    for model in df["model"].unique():
        sub = df[df["model"] == model].groupby("rank")["accuracy"].mean().reset_index()
        ax.plot(sub["rank"], sub["accuracy"], marker="o", label=model, linewidth=2)
    ax.set_title("Accuracy vs Rank (by Model)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 2. Accuracy vs Rank by task
    ax = axes[0, 1]
    for task in df["task"].unique():
        sub = df[df["task"] == task].groupby("rank")["accuracy"].mean().reset_index()
        ax.plot(sub["rank"], sub["accuracy"], marker="s", label=task, linewidth=2)
    ax.set_title("Accuracy vs Rank (by Task)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 3. Speedup factor
    ax = axes[0, 2]
    if "speedup_factor" in df.columns:
        for model in df["model"].unique():
            sub = df[df["model"] == model].groupby("rank")["speedup_factor"].mean()
            ax.plot(sub.index, sub.values, marker="^", label=model, linewidth=2)
        ax.set_title("Speedup vs Rank")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No speedup data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    # 4. Parameter reduction
    ax = axes[1, 0]
    if "param_reduction" in df.columns:
        pd_data = df.groupby(["model", "rank"])["param_reduction"].mean().reset_index()
        models = pd_data["model"].unique()
        width = 0.8 / len(models)
        for i, m in enumerate(models):
            sub = pd_data[pd_data["model"] == m]
            x = sub["rank"] + (i - len(models) / 2) * width
            ax.bar(x, sub["param_reduction"], width=width, label=m)
        ax.set_title("Parameter Reduction")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No param reduction",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    # 5. Accuracy drop distribution
    ax = axes[1, 1]
    if "accuracy_drop" in df.columns:
        drops = df["accuracy_drop"].dropna()
        ax.hist(drops, bins=20, color="lightcoral", edgecolor="darkred", alpha=0.7)
        ax.axvline(
            drops.mean(), color="blue", linestyle="-", label=f"Mean: {drops.mean():.3f}"
        )
        ax.set_title("Accuracy Drop Distribution")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No accuracy drop",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    # 6. Memory reduction
    ax = axes[1, 2]
    if "memory_reduction" in df.columns:
        mem = df.groupby("rank")["memory_reduction"].mean().reset_index()
        ax.bar(
            mem["rank"],
            mem["memory_reduction"],
            color="lightgreen",
            edgecolor="darkgreen",
        )
        ax.set_title("Avg Memory Reduction")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No memory reduction",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    # 7. Accuracy by task type
    ax = axes[2, 0]
    if "task_type" in df.columns:
        for tt in df["task_type"].unique():
            sub = (
                df[df["task_type"] == tt]
                .groupby("rank")["accuracy"]
                .mean()
                .reset_index()
            )
            ax.plot(sub["rank"], sub["accuracy"], marker="D", label=tt, linewidth=2)
        ax.set_title("Accuracy by Task Type")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5, 0.5, "No task type", ha="center", va="center", transform=ax.transAxes
        )
    # 8. Compression ratio heatmap
    ax = axes[2, 1]
    if "compression_ratio" in df.columns:
        pivot = df.pivot_table(
            index="model", columns="rank", values="compression_ratio", aggfunc="mean"
        )
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        plt.colorbar(im, ax=ax, label="Compression Ratio")
        ax.set_title("Compression Ratio Heatmap")
    else:
        ax.text(
            0.5, 0.5, "No comp ratio", ha="center", va="center", transform=ax.transAxes
        )
    # 9. Best config per dataset
    ax = axes[2, 2]
    best = []
    for task in df["task"].unique():
        sub = df[df["task"] == task]
        if "accuracy" in sub.columns:
            idx = sub["accuracy"].idxmax()
            best.append(sub.loc[idx])
    if best:
        best_df = pd.DataFrame(best)
        ax.barh(best_df["task"], best_df["accuracy"], color="skyblue", edgecolor="navy")
        ax.set_title("Best Configuration per Dataset")
        for i, (acc, model, rank) in enumerate(
            zip(best_df["accuracy"], best_df["model"], best_df["rank"])
        ):
            ax.text(acc + 0.005, i, f"{model} r={int(rank)}", va="center", fontsize=8)
    else:
        ax.text(
            0.5, 0.5, "No best config", ha="center", va="center", transform=ax.transAxes
        )
    plt.tight_layout()
    plt.savefig(
        Path(output_dir) / "comprehensive_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_tradeoff(df, output_dir, metric="accuracy") -> None:
    """Plot tradeoff."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    for model in df["model"].unique():
        sub = df[df["model"] == model].dropna(subset=["speedup_factor", metric])
        ax.scatter(sub["speedup_factor"], sub[metric], s=100, label=model, alpha=0.7)
    ax.set_xlabel("Speedup Factor")
    ax.set_ylabel(metric.capitalize())
    ax.set_title("Accuracy vs Speedup Tradeoff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_pareto_front(df, output_dir, x_metric="speedup_factor", y_metric="accuracy") -> None:
    """Plot pareto front."""
    set_style()
    points = df[[x_metric, y_metric, "model", "rank"]].dropna().values
    pareto = []
    for i, p in enumerate(points):
        dominated = False
        for q in points:
            if q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    pareto_df = pd.DataFrame(pareto, columns=[x_metric, y_metric, "model", "rank"])
    fig, ax = plt.subplots(figsize=(10, 8))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        ax.scatter(sub[x_metric], sub[y_metric], label=model, alpha=0.5)
    ax.scatter(
        pareto_df[x_metric],
        pareto_df[y_metric],
        color="red",
        s=150,
        marker="*",
        label="Pareto front",
    )
    ax.set_xlabel(x_metric.replace("_", " ").capitalize())
    ax.set_ylabel(y_metric.capitalize())
    ax.set_title("Pareto Frontier of Configurations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "pareto_front.png", dpi=150)
    plt.close()


def plot_rank_allocation(recommended_ranks, output_dir) -> None:
    """Plot rank allocation."""
    set_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = list(range(len(recommended_ranks)))
    ax.bar(layers, recommended_ranks, color="steelblue", edgecolor="navy")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Recommended Rank")
    ax.set_title("Layer‑wise Rank Allocation Recommendation")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "rank_allocation.png", dpi=150)
    plt.close()


def plot_training_history(history: dict, save_path: str) -> None:
    """Plot training history."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if history.get("train_steps") and history.get("train_loss"):
        ax.plot(
            history["train_steps"],
            history["train_loss"],
            label="Train Loss",
            color="blue",
        )
    if history.get("eval_steps") and history.get("eval_loss"):
        ax.plot(
            history["eval_steps"],
            history["eval_loss"],
            label="Eval Loss",
            color="red",
            marker="o",
        )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


plot_loss_curves = plot_loss_curve
plot_accuracy_curves = plot_accuracy_curve
plot_gradient_flow = plot_training_time
