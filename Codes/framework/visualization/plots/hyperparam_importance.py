"""Hyperparameter importance (fANOVA-like) based on correlation with metric."""
from typing import Any

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_hyperparam_importance(history, metrics, config, output_dir, opts) -> Any:
    """Plot hyperparam importance."""
    runs_dir = Path(output_dir).parent.parent
    run_dirs = sorted(runs_dir.glob("*/artifacts/final_metrics.json"))
    if len(run_dirs) < 10:
        logger.info("Need at least 10 runs for hyperparameter importance.")
        return []
    records = []
    for rp in run_dirs:
        try:
            with open(rp) as f:
                fm = json.load(f)
            cfg_path = rp.parent / "config.json"
            if not cfg_path.exists():
                continue
            with open(cfg_path) as f:
                cfg = json.load(f)
            records.append(
                {
                    "accuracy": fm.get("eval_accuracy", 0),
                    "lr": float(cfg.get("learning_rate", cfg.get("lr", 0))),
                    "batch_size": float(cfg.get("batch_size", 1)),
                    "epochs": float(cfg.get("epochs", 1)),
                    "max_len": float(cfg.get("max_seq_length", 1)),
                    "warmup": float(cfg.get("warmup_steps", 0)),
                }
            )
        except Exception:
            continue
    if len(records) < 10:
        return []
    df = pd.DataFrame(records)
    corrs = {}
    for col in ["lr", "batch_size", "epochs", "max_len", "warmup"]:
        if col in df.columns:
            corrs[col] = abs(df[col].corr(df["accuracy"]))
    sorted_items = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 5]))
    names = [x[0] for x in sorted_items]
    vals = [x[1] for x in sorted_items]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(vals)))
    ax.barh(names, vals, color=colors, edgecolor="white")
    ax.set_xlabel("|Correlation| with Accuracy")
    ax.set_title("Hyperparameter Importance")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = str(Path(output_dir) / "hyperparam_importance.png")
    fig.savefig(out, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [out]
