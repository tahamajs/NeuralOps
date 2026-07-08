import matplotlib.pyplot as plt


def plot_gradient_flow(history, metrics, config, output_dir, opts):
    # Expects 'grad_norm' entries in history
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
    path = output_dir / "gradient_flow.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [str(path)]
