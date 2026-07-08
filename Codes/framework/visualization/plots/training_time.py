from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_time(history, metrics, config, output_dir, opts):
    # Try to use 'elapsed' field if logged, else estimate
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
    plt.close(fig)
    return [str(path)]
