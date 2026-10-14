"""
Tests for Needle-In-A-Haystack (NIAH) and LM Evaluation Harness Adapter.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from neuralops.evaluation.niah_passkey import (
    NeedleInAHaystackEvaluator,
    plot_niah_heatmap,
)
from neuralops.evaluation.lm_eval_adapter import (
    NeuralOpsLMEvalAdapter,
    evaluate_zero_shot_tasks,
)


class MockTokenizer:
    """Lightweight deterministic tokenizer for fast testing without HuggingFace downloads."""
    def __init__(self):
        self.vocab = {f"word_{i}": i for i in range(100)}
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Hash each whitespace word to token id
        words = text.strip().split()
        return [abs(hash(w)) % 200 for w in words] or [1]

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join([f"tok_{t}" for t in token_ids])


def test_niah_passkey_evaluator_and_heatmap(tmp_path: Path):
    """Verify NeedleInAHaystackEvaluator runs across 2D grid and plots heatmap."""
    config = GPT2Config(
        vocab_size=250,
        n_positions=256,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    model = GPT2LMHeadModel(config).eval()
    tokenizer = MockTokenizer()

    evaluator = NeedleInAHaystackEvaluator(
        model=model,
        tokenizer=tokenizer,
        context_lengths=[64, 128],
        needle_depths=[0.2, 0.8],
        passkey_digits=3,
        device="cpu",
    )
    results = evaluator.evaluate_grid(num_trials_per_cell=1)

    assert "accuracy_grid" in results
    assert len(results["accuracy_grid"]) == 2
    assert len(results["accuracy_grid"][0]) == 2
    assert 0.0 <= results["mean_retrieval_accuracy"] <= 1.0

    # Test plot_niah_heatmap
    plot_path = tmp_path / "test_niah.png"
    plot_niah_heatmap(results, plot_path)
    assert plot_path.exists()
    assert (tmp_path / "test_niah.pdf").exists()


def test_lm_eval_adapter_loglikelihood_and_generate():
    """Verify NeuralOpsLMEvalAdapter interface."""
    config = GPT2Config(
        vocab_size=250,
        n_positions=256,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    model = GPT2LMHeadModel(config).eval()
    tokenizer = MockTokenizer()

    adapter = NeuralOpsLMEvalAdapter(model, tokenizer, device="cpu")

    # 1. loglikelihood
    reqs = [
        ("The sky is", " blue"),
        ("Artificial intelligence is", " transformative"),
    ]
    ll_results = adapter.loglikelihood(reqs)
    assert len(ll_results) == 2
    for ll, is_greedy in ll_results:
        assert isinstance(ll, float)
        assert isinstance(is_greedy, bool)

    # 2. loglikelihood_rolling
    rolling = adapter.loglikelihood_rolling(["Hello world sentence.", "Another test text."])
    assert len(rolling) == 2

    # 3. generate_until
    gen_reqs = [("Prompt:", {"max_new_tokens": 5, "until": ["\n"]})]
    gen_res = adapter.generate_until(gen_reqs)
    assert len(gen_res) == 1
    assert isinstance(gen_res[0], str)

    # 4. Zero-shot benchmark helper
    scores = evaluate_zero_shot_tasks(model, tokenizer, tasks=("piqa",), device="cpu")
    assert "piqa" in scores
