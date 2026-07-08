import logging

logger = logging.getLogger(__name__)


def plot_parameter_distribution(history, metrics, config, output_dir, opts):
    # This plot needs the model object, which is not available in history.
    # Store the parameter count as a JSON file during training and read it here.
    param_path = output_dir.parent / "artifacts" / "param_counts.json"
    if not param_path.exists():
        logger.info("Parameter count plot requested but no param_counts.json saved.")
        return []
    import json

    import matplotlib.pyplot as plt

    with open(param_path) as f:
        param_data = json.load(f)
    fig, ax = plt.subplots(figsize=opts.get("figsize", [8, 6]))
    layers = list(param_data.keys())
    counts = [param_data[l] for l in layers]
    ax.barh(layers, counts, color="#21908C")
    ax.set_xlabel("Parameter Count")
    ax.set_title("Parameter Distribution per Layer")
    plt.tight_layout()
    path = output_dir / "parameter_count.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [str(path)]
