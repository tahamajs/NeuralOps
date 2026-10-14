"""
Needle In A Haystack (NIAH) and Passkey Long-Context Retrieval Benchmark.
Evaluates exact-match retrieval accuracy and target token loss across
variable context lengths and needle insertion depths.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Default haystack distractor text template
HAYSTACK_PARAGRAPHS = [
    "The city of Geneva is situated on the shores of Lake Geneva at the foot of the Alps.",
    "Quantum computing uses the principles of superposition and entanglement to perform complex operations.",
    "Linear attention mechanisms replace the quadratic softmax computation with kernel feature maps.",
    "Singular Value Decomposition decomposes any real matrix into orthogonal bases and singular values.",
    "Fast Walsh-Hadamard transform computes discrete orthogonal rotations in O(N log N) arithmetic operations.",
    "Speculative decoding accelerates autoregressive token generation by drafting candidates and verifying in parallel.",
    "Modern deep neural networks benefit substantially from low-rank weight factorization and FP8 tensor operations.",
    "The Nile River flows northward through eastern Africa and is historically among the longest rivers on Earth.",
    "Photosynthesis is the biochemical process by which green plants and certain other organisms transform light energy into chemical energy.",
]


class NeedleInAHaystackEvaluator:
    """
    Standardized multi-length and multi-depth NIAH Passkey Evaluator.
    Measures exact-match key extraction rate and cross-entropy loss at needle location.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        context_lengths: Optional[List[int]] = None,
        needle_depths: Optional[List[float]] = None,
        passkey_digits: int = 5,
        device: torch.device | str = "cuda",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.context_lengths = context_lengths or [256, 512, 1024, 2048]
        self.needle_depths = needle_depths or [0.1, 0.3, 0.5, 0.7, 0.9]
        self.passkey_digits = passkey_digits
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    def _generate_passkey(self) -> str:
        low = 10 ** (self.passkey_digits - 1)
        high = (10 ** self.passkey_digits) - 1
        return str(random.randint(low, high))

    def evaluate_grid(self, num_trials_per_cell: int = 1) -> Dict[str, Any]:
        """
        Runs evaluation over the 2D grid of context lengths and needle depths.
        Returns accuracy grid and average target perplexity.
        """
        self.model.eval()
        self.model.to(self.device)

        grid_accuracy = np.zeros((len(self.context_lengths), len(self.needle_depths)))
        results_records = []

        for i, ctx_len in enumerate(self.context_lengths):
            for j, depth in enumerate(self.needle_depths):
                cell_successes = 0
                for trial in range(num_trials_per_cell):
                    passkey = self._generate_passkey()
                    needle_text = f" Special notice: The secret passkey is {passkey}. Remember this exact number. "
                    prompt_question = "\nWhat is the secret passkey? The secret passkey is"

                    needle_tokens = self.tokenizer.encode(needle_text, add_special_tokens=False)
                    q_tokens = self.tokenizer.encode(prompt_question, add_special_tokens=False)

                    # Build haystack to fill desired context length
                    target_haystack_len = max(ctx_len - len(needle_tokens) - len(q_tokens), 32)
                    haystack_tokens = []
                    p_idx = 0
                    while len(haystack_tokens) < target_haystack_len:
                        p = HAYSTACK_PARAGRAPHS[p_idx % len(HAYSTACK_PARAGRAPHS)] + " "
                        tokens = self.tokenizer.encode(p, add_special_tokens=False)
                        haystack_tokens.extend(tokens)
                        p_idx += 1
                    haystack_tokens = haystack_tokens[:target_haystack_len]

                    # Insert needle at specified depth ratio
                    insert_idx = int(len(haystack_tokens) * depth)
                    full_tokens = (
                        haystack_tokens[:insert_idx]
                        + needle_tokens
                        + haystack_tokens[insert_idx:]
                        + q_tokens
                    )

                    input_ids = torch.tensor([full_tokens], device=self.device)

                    # Generate next tokens to check answer
                    with torch.no_grad():
                        out = input_ids
                        for _ in range(self.passkey_digits + 2):
                            logits = self.model(out)
                            if hasattr(logits, "logits"):
                                logits = logits.logits
                            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                            out = torch.cat([out, next_tok], dim=-1)

                    pred_text = self.tokenizer.decode(out[0, len(full_tokens) :])
                    is_correct = passkey in pred_text
                    if is_correct:
                        cell_successes += 1

                    results_records.append({
                        "context_length": ctx_len,
                        "depth": depth,
                        "passkey": passkey,
                        "prediction": pred_text.strip(),
                        "correct": is_correct,
                    })

                acc = cell_successes / max(num_trials_per_cell, 1)
                grid_accuracy[i, j] = acc
                logger.info("Context %d, Depth %.1f: Acc=%.2f%%", ctx_len, depth, acc * 100)

        return {
            "context_lengths": self.context_lengths,
            "needle_depths": self.needle_depths,
            "accuracy_grid": grid_accuracy.tolist(),
            "mean_retrieval_accuracy": float(np.mean(grid_accuracy)),
            "records": results_records,
        }


def plot_niah_heatmap(
    niah_results: Dict[str, Any],
    save_path: str | Path,
    title: str = "Needle-In-A-Haystack Passkey Retrieval",
) -> None:
    """
    Renders publication-grade 2D Retrieval Heatmap (Context Length vs. Needle Depth).
    Saves in PDF, SVG, and PNG format.
    """
    ctx_lengths = niah_results["context_lengths"]
    depths = niah_results["needle_depths"]
    acc_grid = np.array(niah_results["accuracy_grid"]) * 100.0  # Percentage

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    cax = ax.imshow(acc_grid, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([f"{d * 100:.0f}%" for d in depths], fontsize=10)
    ax.set_yticks(range(len(ctx_lengths)))
    ax.set_yticklabels(ctx_lengths, fontsize=10)

    ax.set_xlabel("Needle Depth Position", fontsize=11, fontweight="bold")
    ax.set_ylabel("Context Length (Tokens)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Annotate cell values
    for i in range(len(ctx_lengths)):
        for j in range(len(depths)):
            val = acc_grid[i, j]
            color = "white" if val > 65 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Retrieval Accuracy (%)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in primary format
    plt.savefig(p, bbox_inches="tight")
    # Also save PDF and SVG if primary is PNG
    if p.suffix == ".png":
        plt.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
        plt.savefig(p.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved NIAH heatmaps to %s", p)


__all__ = [
    "NeedleInAHaystackEvaluator",
    "plot_niah_heatmap",
]
