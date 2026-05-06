"""Residual / prediction error plots for regression tasks (e.g., STS‑B)."""
import json, logging, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)

def plot_residuals(history, metrics, config, output_dir, opts):
    preds_path = Path(output_dir).parent / "artifacts" / "eval_predictions.json"
    if not preds_path.exists():
        logger.info("Residual plot requires eval_predictions.json")
        return []
    with open(preds_path) as f:
        data = json.load(f)
    labels = np.array(data.get("labels", []))
    preds  = np.array(data.get("predictions_continuous", data.get("predictions", [])))
    if len(labels) == 0:
        return []
    # Ensure we have continuous predictions (regression)
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds = preds[:, 0]  # fallback
    residuals = labels - preds

    fig, axes = plt.subplots(1, 3, figsize=opts.get("figsize", [18, 5]))
    # 1. Residuals vs. Predicted
    ax = axes[0]
    ax.scatter(preds, residuals, alpha=0.5, edgecolors='k', linewidth=0.3, color='#3B528B')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs. Predicted'); ax.grid(True, alpha=0.3)

    # 2. Histogram of residuals
    ax = axes[1]
    ax.hist(residuals, bins=30, color='#21908C', edgecolor='white', alpha=0.8)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Residual'); ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution'); ax.grid(True, axis='y', alpha=0.3)

    # 3. Q-Q plot
    ax = axes[2]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)')
    ax.get_lines()[0].set_color('#3B528B')
    ax.get_lines()[1].set_color('red')

    plt.tight_layout()
    out = str(Path(output_dir) / "residual_analysis.png")
    fig.savefig(out, dpi=opts.get("dpi", 150), bbox_inches='tight')
    plt.close(fig)
    return [out]