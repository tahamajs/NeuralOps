import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

def plot_loss_curve(history, metrics, config, output_dir, opts):
    """Plot smoothed training / evaluation loss curves."""
    steps = [h.get('step', i) for i, h in enumerate(history) if 'loss' in h]
    train_loss = [h['loss'] for h in history if 'loss' in h]
    eval_steps = [h.get('step', i) for i, h in enumerate(history) if 'eval_loss' in h]
    eval_loss = [h['eval_loss'] for h in history if 'eval_loss' in h]

    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 6]))
    smooth = opts.get("loss_smooth", 0.6)
    if smooth > 0 and len(train_loss) > 5:
        window = max(1, int(len(train_loss) * smooth / 10))
        train_loss = uniform_filter1d(train_loss, size=window)
    ax.plot(steps, train_loss, label='Train Loss', color='#3B528B')
    if eval_steps:
        ax.plot(eval_steps, eval_loss, 'o-', label='Eval Loss', color='#21908C', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "loss_curve.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches='tight')
    plt.close(fig)
    return [str(path)]