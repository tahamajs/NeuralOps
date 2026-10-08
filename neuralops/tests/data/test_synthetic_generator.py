"""Tests for neuralops.data.synthetic_generator module."""

import pytest
import torch
from collections import Counter

from neuralops.data.synthetic_generator import (
    SyntheticClassificationDataset,
    SyntheticPerplexityDataset,
    compute_empirical_entropy,
    generate_zipfian_markov_tokens,
    generate_zipfian_tokens,
)


class TestZipfianTokenGenerators:
    """Test Zipfian power-law distributed token sequence generators."""

    def test_generate_zipfian_tokens_shape(self):
        tokens = generate_zipfian_tokens(
            num_samples=16,
            seq_len=64,
            vocab_size=1000,
            seed=42,
        )
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (16, 64)
        assert (tokens >= 0).all()
        assert (tokens < 1000).all()

    def test_generate_zipfian_markov_tokens(self):
        tokens = generate_zipfian_markov_tokens(
            num_samples=8,
            seq_len=32,
            vocab_size=500,
            zipf_param=1.2,
            seed=42,
        )
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (8, 32)

    def test_zipfian_distribution_non_uniform(self):
        """Verify Zipfian distribution is non-uniform and heavily biased toward low indices."""
        tokens = generate_zipfian_tokens(num_samples=100, seq_len=100, vocab_size=200, seed=42)
        flat = tokens.flatten().tolist()
        counts = Counter(flat)
        sorted_counts = sorted(counts.values(), reverse=True)
        # Top rank frequency is much higher than median frequency
        assert sorted_counts[0] > 3 * sorted_counts[len(sorted_counts) // 2]

    def test_deterministic_generation(self):
        t1 = generate_zipfian_tokens(num_samples=10, seq_len=20, vocab_size=500, seed=123)
        t2 = generate_zipfian_tokens(num_samples=10, seq_len=20, vocab_size=500, seed=123)
        assert torch.equal(t1, t2)


class TestSyntheticDatasets:
    """Test PyTorch SyntheticPerplexityDataset and SyntheticClassificationDataset."""

    def test_perplexity_dataset(self):
        ds = SyntheticPerplexityDataset(
            num_samples=10,
            seq_len=32,
            vocab_size=100,
            seed=42,
        )
        assert len(ds) == 10
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].shape == (32,)

    def test_classification_dataset(self):
        ds = SyntheticClassificationDataset(
            num_samples=12,
            seq_len=16,
            vocab_size=100,
            num_classes=2,
            seed=42,
        )
        assert len(ds) == 12
        item = ds[0]
        assert "input_ids" in item
        assert "labels" in item
        assert item["labels"].item() in (0, 1)

    def test_compute_empirical_entropy(self):
        tokens = torch.randint(0, 50, (5, 32))
        entropy = compute_empirical_entropy(tokens, vocab_size=50)
        assert isinstance(entropy, float)
        assert entropy >= 0.0
