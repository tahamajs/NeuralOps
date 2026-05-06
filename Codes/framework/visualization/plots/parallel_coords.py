"""Parallel coordinates for hyperparameter optimization runs."""
import json, logging, numpy as np, matplotlib.pyplot as plt, pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def plot_parallel_coords(history, metrics, config, output_dir, opts):
    """Reads all runs from the parent directory and plots parallel coordinates."""
    runs_dir = Path(output_dir).parent.parent  # go up to runs root
    run_dirs = sorted(runs_dir.glob("*/artifacts/final_metrics.json"))
    if len(run_dirs) < 2:
        logger.info("Need at least 2 completed runs for parallel coordinates.")
        return []

    records = []
    for rp in run_dirs:
        try:
            with open(rp) as f:
                fm = json.load(f)
            config_path = rp.parent / "config.json"
            cfg = {}
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
            record = {
                "accuracy": fm.get("eval_accuracy", 0),
                "loss": fm.get("eval_loss", 0),
                "lr": cfg.get("learning_rate", cfg.get("lr", 0)),
                "batch_size": cfg.get("batch_size", 0),
                "epochs": cfg.get("epochs", 0),
                "max_len": cfg.get("max_seq_length", 0),
                "run_id": rp.parent.parent.name,
            }
            records.append(record)
        except Exception:
            continue
    if len(records) < 2:
        return []

    df = pd.DataFrame(records)
    metric_col = "accuracy"
    norm = plt.Normalize(df[metric_col].min(), df[metric_col].max())
    colors = plt.cm.viridis(norm(df[metric_col]))

    fig, ax = plt.subplots(figsize=opts.get("figsize", [14, 6]))
    cols = ["lr", "batch_size", "epochs", "max_len", "accuracy"]
    for i, row in df.iterrows():
        ax.plot(range(len(cols)), [row[c] for c in cols], 'o-', color=colors[i],
                linewidth=1, markersize=6, alpha=0.7)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_title('Parallel Coordinates – Hyperparameter Runs')
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, label=metric_col)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = str(Path(output_dir) / "parallel_coordinates.png")
    fig.savefig(out, dpi=opts.get("dpi", 150), bbox_inches='tight')
    plt.close(fig)
    return [out]