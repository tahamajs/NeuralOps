"""UMAP / t-SNE embedding visualization of hidden states or predictions."""
from typing import Any

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_embedding(history, metrics, config, output_dir, opts) -> Any:
    """Plot embedding."""
    emb_path = Path(output_dir).parent / "artifacts" / "embeddings.json"
    if not emb_path.exists():
        logger.info("Embedding plot requires embeddings.json (hidden states).")
        return []
    with open(emb_path) as f:
        data = json.load(f)
    vectors = np.array(data.get("vectors", []))
    labels = np.array(data.get("labels", []))
    data.get("texts", [])
    if len(vectors) == 0:
        return []

    method = opts.get("embedding_method", "umap")
    try:
        if method == "tsne":
            from sklearn.manifold import TSNE

            reducer = TSNE(
                n_components=2, random_state=42, perplexity=min(30, len(vectors) - 1)
            )
        else:
            from umap import UMAP

            reducer = UMAP(n_components=2, random_state=42)
    except ImportError:
        logger.info(f"{method} not installed. Run: pip install {method}-learn")
        return []

    embedded = reducer.fit_transform(vectors)
    unique_labels = np.unique(labels)
    cmap = plt.cm.tab10 if len(unique_labels) <= 10 else plt.cm.tab20

    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 8]))
    scatter = ax.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=labels,
        cmap=cmap,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.3,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Class")
    ax.set_title(f"{method.upper()} Projection of Hidden States")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    out = str(Path(output_dir) / f"embedding_{method}.png")
    fig.savefig(out, dpi=opts.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    return [out]
