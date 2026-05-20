"""Comparison‑specific plotting: critical difference diagram and pairwise heatmap."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple

def plot_critical_difference_diagram(
    groups: Dict[str, np.ndarray],
    alpha: float = 0.05,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4),
) -> Optional[str]:
    names = list(groups.keys())
    n = len(names)
    if n < 2:
        return None

    arrays = [groups[k] for k in names]
    N = min(len(a) for a in arrays)
    if N < 2:
        return None

    from scipy import stats as sp_stats
    from scipy.stats import studentized_range

    ranks = np.zeros((N, n))
    for i in range(N):
        row = [arrays[j][i] for j in range(n)]
        ranks[i] = sp_stats.rankdata(row)
    avg_ranks = np.mean(ranks, axis=0)

    sorted_idx = np.argsort(avg_ranks)
    sorted_names = [names[i] for i in sorted_idx]
    sorted_ranks = avg_ranks[sorted_idx]

    try:
        q_alpha = studentized_range.ppf(1 - alpha, n, np.inf)
    except Exception:
        q_alpha = 3.0
    cd = q_alpha * np.sqrt(n * (n + 1) / (6 * N))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(min(sorted_ranks) - 1, max(sorted_ranks) + 1)
    ax.set_ylim(0, 1.5)

    for i in range(n - 1):
        if abs(sorted_ranks[i + 1] - sorted_ranks[i]) < cd:
            ax.plot([sorted_ranks[i], sorted_ranks[i + 1]], [0.3, 0.3], 'k-', linewidth=3)

    for i, (name, rank) in enumerate(zip(sorted_names, sorted_ranks)):
        ax.plot(rank, 0.3, 'o', markersize=15, markerfacecolor='white',
                markeredgecolor='black', markeredgewidth=2)
        ax.text(rank, 0.15, f"{rank:.2f}", ha='center', va='center', fontsize=9)
        ax.text(rank, -0.2, name, ha='center', va='top', fontsize=11,
                rotation=30 if len(name) > 10 else 0)

    ax.set_title(f'Critical Difference Diagram (α={alpha})')
    ax.axis('off')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    return None


def plot_pairwise_comparison_heatmap(
    groups: Dict[str, np.ndarray],
    metric: str = "Accuracy",
    p_values: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> Optional[str]:
    names = sorted(groups.keys())
    n = len(names)
    if n < 2:
        return None

    means = [np.mean(groups[k]) for k in names]
    diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff_matrix[i, j] = means[j] - means[i]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-np.max(np.abs(diff_matrix)), vmax=np.max(np.abs(diff_matrix)))
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{diff_matrix[i, j]:.3f}", ha='center', va='center',
                    fontsize=9, color='white' if abs(diff_matrix[i, j]) > np.max(np.abs(diff_matrix))/2 else 'black')
    plt.colorbar(im, ax=ax, label=f'{metric} Difference')
    ax.set_title(f'Pairwise Mean {metric} Differences')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_path
    return None