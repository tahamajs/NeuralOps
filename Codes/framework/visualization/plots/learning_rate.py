from pathlib import Path

import matplotlib.pyplot as plt


def plot_lr_schedule(history, metrics, config, output_dir, opts):
    steps = [h.get("step", i) for i, h in enumerate(history) if "learning_rate" in h]
    lr = [h["learning_rate"] for h in history if "learning_rate" in h]
    if not steps:
        return []
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 4]))
    ax.plot(steps, lr, color="#FDE725")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "lr_schedule.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [str(path)]
