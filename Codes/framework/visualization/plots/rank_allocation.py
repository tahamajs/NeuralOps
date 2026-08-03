import matplotlib.pyplot as plt


def plot_rank_allocation(history, metrics, config, output_dir, opts):
    # Expects per‑layer rank information stored during training in a JSON file
    """Plot rank allocation."""
    rank_path = output_dir.parent / "artifacts" / "rank_allocation.json"
    if not rank_path.exists():
        return []
    import json

    with open(rank_path) as f:
        rank_data = json.load(f)
    layers = list(rank_data.keys())
    ranks = [rank_data[l] for l in layers]
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 4]))
    ax.bar(layers, ranks, color="#5DC963")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Selected Rank")
    ax.set_title("Rank Allocation per Layer")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "rank_allocation.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [str(path)]
