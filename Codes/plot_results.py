import json
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"
TASKS = ["cola", "mrpc", "rte", "stsb", "sst2", "qnli", "qqp", "mnli"]

def extract_score(task, metrics):
    if task == "stsb":
        return metrics.get("eval_spearmanr", metrics.get("eval_pearson", 0.0))
    else:
        return metrics.get("eval_accuracy", 0.0)

def plot_all():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, task in enumerate(TASKS):
        with open(f"{RESULTS_DIR}/{task}/summary.json") as f:
            data = json.load(f)

        ranks = []
        scores = []
        for rank_str, metrics in data.items():
            if rank_str == "None":
                continue
            ranks.append(int(rank_str))
            scores.append(extract_score(task, metrics))

        sorted_pairs = sorted(zip(ranks, scores))
        if sorted_pairs:
            ranks, scores = zip(*sorted_pairs)
            axes[i].plot(ranks, scores, marker='o', linestyle='-', linewidth=2)
        axes[i].set_title(task.upper())
        axes[i].set_xlabel("Rank (r)")
        axes[i].set_ylabel("Score (Acc / Spearman)")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/accuracy_vs_rank.png", dpi=150)
    plt.close()
    print(f"Plot saved as {RESULTS_DIR}/accuracy_vs_rank.png")

if __name__ == "__main__":
    plot_all()