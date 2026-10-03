"""
High-Fidelity Synthetic Dataset & Token Generator for NLP / LLM Evaluation.

Replaces uniform random token noise with a power-law Zipfian distribution:
$$P(k) = \\frac{1 / k^s}{\\sum_{n=1}^V 1 / n^s} \\quad \\text{with } s \\approx 1.15$$
to simulate authentic natural language entropy, word rank frequency curves,
and valid causal perplexity evaluation dynamics.
"""

from __future__ import annotations

import logging
import random
import re
import string
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def generate_zipfian_tokens(
    num_samples: int,
    seq_len: int,
    vocab_size: int = 32000,
    zipf_param: float = 1.15,
    seed: int = 42,
    min_token_id: int = 1,
) -> torch.Tensor:
    """
    Generates token sequences following a Zipfian power-law distribution.
    
    Ensures that low-index tokens (common stop words, articles) occur with high frequency
    while high-index tokens (rare technical terms) occur sparsely, mirroring natural language.

    Args:
        num_samples: Number of sequences to generate.
        seq_len: Sequence length (number of tokens per sample).
        vocab_size: Size of vocabulary.
        zipf_param: Power-law exponent $s > 0$ (default $1.15$, typical for natural language).
        seed: Random seed for deterministic reproducibility.
        min_token_id: Minimum token ID offset (default 1 to reserve 0 for PAD).

    Returns:
        torch.Tensor: Long tensor of shape `(num_samples, seq_len)` containing sampled token IDs.
    """
    np_rng = np.random.default_rng(seed)

    effective_vocab = max(2, vocab_size - min_token_id)

    # FIX: Compute exact Zipfian rank probabilities: P(k) = (1 / k^s) / zeta(s)
    # to simulate authentic natural language entropy and prevent flat uniform noise distortion.
    ranks = np.arange(1, effective_vocab + 1, dtype=np.float64)
    probs = 1.0 / (ranks ** zipf_param)
    probs_sum = np.sum(probs)
    if probs_sum > 0:
        probs /= probs_sum
    else:
        probs = np.full(effective_vocab, 1.0 / effective_vocab)

    # Sample token indices from power-law distribution
    flat_tokens = np_rng.choice(
        effective_vocab, size=(num_samples, seq_len), p=probs, replace=True
    )

    # Shift indices to respect min_token_id (reserving special tokens)
    token_tensor = torch.from_numpy(flat_tokens + min_token_id).long()
    return token_tensor


def generate_zipfian_markov_tokens(
    num_samples: int,
    seq_len: int,
    vocab_size: int = 32000,
    zipf_param: float = 1.15,
    transition_temp: float = 0.8,
    seed: int = 42,
    min_token_id: int = 1,
) -> torch.Tensor:
    """
    Generates token sequences with Zipfian marginal rank distribution combined with
    first-order Markov sequential transition coupling.

    Args:
        num_samples: Number of sequences.
        seq_len: Sequence length.
        vocab_size: Vocabulary size.
        zipf_param: Zipf exponent.
        transition_temp: Softmax temperature for Markov transition matrix.
        seed: Random seed.
        min_token_id: Minimum token ID offset.

    Returns:
        torch.Tensor: Long tensor of shape `(num_samples, seq_len)`.
    """
    np_rng = np.random.default_rng(seed)
    effective_vocab = max(2, min(vocab_size - min_token_id, 2048))

    # FIX: Compute Zipfian unigram prior
    ranks = np.arange(1, effective_vocab + 1, dtype=np.float64)
    unigram_probs = 1.0 / (ranks ** zipf_param)
    unigram_probs /= unigram_probs.sum()

    tokens = np.zeros((num_samples, seq_len), dtype=np.int64)
    tokens[:, 0] = np_rng.choice(effective_vocab, size=num_samples, p=unigram_probs)

    for t in range(1, seq_len):
        prev = tokens[:, t - 1]
        # Transition distribution biased by previous token and global Zipfian prior
        next_choices = np_rng.choice(effective_vocab, size=num_samples, p=unigram_probs)
        # 30% local Markov continuity, 70% Zipfian base distribution
        mask = np_rng.random(size=num_samples) < 0.3
        local_step = (prev + np_rng.integers(-3, 4, size=num_samples)) % effective_vocab
        tokens[:, t] = np.where(mask, local_step, next_choices)

    return torch.from_numpy(tokens + min_token_id).long()


def compute_empirical_entropy(tokens: torch.Tensor, vocab_size: int = 32000) -> float:
    """
    Computes the empirical Shannon entropy (in bits) of a tokenized tensor.

    Args:
        tokens: Long tensor of token IDs.
        vocab_size: Total vocabulary size.

    Returns:
        float: Shannon entropy $H(X) = -\\sum p_i \\log_2(p_i)$ in bits.
    """
    flat = tokens.view(-1).cpu().numpy()
    counts = np.bincount(flat, minlength=vocab_size)
    counts = counts[counts > 0]
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


class SyntheticPerplexityDataset(Dataset):
    """
    Synthetic Language Modeling Dataset using Zipfian power-law token distribution
    for authentic causal perplexity evaluation without empirical artifact distortion.
    """

    def __init__(
        self,
        num_samples: int = 128,
        seq_len: int = 128,
        vocab_size: int = 32000,
        zipf_param: float = 1.15,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # FIX: Generate Zipfian tokens to reflect natural language entropy
        self.input_ids = generate_zipfian_tokens(
            num_samples=num_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            zipf_param=zipf_param,
            seed=seed,
        )
        self.attention_mask = torch.ones((num_samples, seq_len), dtype=torch.long)
        self.labels = self.input_ids.clone()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


# RESEARCH FIX: Compatibility alias for language modeling datasets
SyntheticLanguageModelingDataset = SyntheticPerplexityDataset


class SyntheticClassificationDataset(Dataset):
    """
    High-fidelity synthetic classification dataset using Zipfian token distributions
    and class-conditional semantic bias for realistic classifier benchmarking.
    """

    def __init__(
        self,
        num_samples: int = 64,
        seq_len: int = 32,
        vocab_size: int = 1000,
        num_classes: int = 2,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        # FIX: Generate Zipfian base tokens
        self.input_ids = generate_zipfian_tokens(
            num_samples=num_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            seed=seed,
        )
        self.attention_mask = torch.ones((num_samples, seq_len), dtype=torch.long)

        # Generate realistic balanced labels
        rng = np.random.default_rng(seed)
        self.labels = torch.from_numpy(
            rng.integers(0, num_classes, size=(num_samples,))
        ).long()

        # Inject class-conditional salient tokens to simulate real sentiment / classification signal
        for i in range(num_samples):
            label_val = self.labels[i].item()
            salient_token = 10 + label_val * 5
            pos = rng.integers(1, seq_len - 1)
            self.input_ids[i, pos] = salient_token

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


DEFAULT_DOMAINS: dict[str, dict[str, list[str]]] = {
    "movies": {
        "pos_adj": ["exceptional", "phenomenal", "brilliant", "masterful", "captivating"],
        "neg_adj": ["disastrous", "tedious", "bland", "lifeless", "confusing"],
        "verbs": ["delivered", "crafted", "executed", "failed", "collapsed"],
        "nouns": ["narrative", "performance", "cinematography", "script", "direction"],
    },
    "tech": {
        "pos_adj": ["incredible", "seamless", "intuitive", "powerful", "lightning-fast"],
        "neg_adj": ["buggy", "slow", "unreliable", "clunky", "overpriced"],
        "verbs": ["works", "performs", "loads", "crashes", "freezes"],
        "nouns": ["interface", "performance", "battery life", "screen quality", "processing speed"],
    },
    "restaurants": {
        "pos_adj": ["delicious", "amazing", "flavorful", "fresh", "authentic"],
        "neg_adj": ["terrible", "bland", "overpriced", "greasy", "stale"],
        "verbs": ["served", "prepared", "cooked", "presented", "plated"],
        "nouns": ["food", "service", "atmosphere", "presentation", "value"],
    },
    "books": {
        "pos_adj": ["engaging", "thought-provoking", "well-written", "captivating", "insightful"],
        "neg_adj": ["boring", "poorly-written", "tedious", "confusing", "disappointing"],
        "verbs": ["writes", "describes", "narrates", "develops", "concludes"],
        "nouns": ["plot", "character", "writing style", "pacing", "dialogue"],
    },
}

TEMPLATES_POS: list[str] = [
    "The {adj} {noun} was {adv} {verb}.",
    "I found the {noun} to be {adj} and {adv} {verb}.",
    "The {adj} {noun} {prep_phrase}.",
    "What an {adj} {noun}! It {verb} {adv}.",
]

TEMPLATES_NEG: list[str] = [
    "The {adj} {noun} was {adv} {verb}.",
    "I found the {noun} to be {adj} and {adv} {verb}.",
    "Unfortunately, the {adj} {noun} {prep_phrase}.",
    "What a {adj} {noun}! It {verb} {adv}.",
]


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text[0].upper() + text[1:] if text else text
    if text and text[-1] not in string.punctuation:
        text += "."
    return text


def _generate_batch(domain_key: str, size: int, label: int, is_positive: bool) -> list[dict[str, Any]]:
    vocab = DEFAULT_DOMAINS.get(domain_key, DEFAULT_DOMAINS["movies"])
    templates = TEMPLATES_POS if is_positive else TEMPLATES_NEG
    prep_phrases = (
        ["with remarkable precision", "with incredible attention to detail", "in every aspect"]
        if is_positive
        else ["with terrible execution", "with poor craftsmanship", "in many ways"]
    )
    intensifiers = ["extremely", "highly", "deeply", "profoundly", "remarkably"]

    data = []
    for _ in range(size):
        adj = random.choice(vocab["pos_adj" if is_positive else "neg_adj"])
        noun = random.choice(vocab["nouns"])
        verb = random.choice(vocab["verbs"])
        adv = "perfectly" if is_positive else "terribly"
        prep = random.choice(prep_phrases)
        intens = random.choice(intensifiers) if random.random() > 0.7 else ""
        if intens:
            adj = f"{intens} {adj}"
        text = random.choice(templates).format(
            adj=adj, noun=noun, verb=verb, adv=adv, prep_phrase=prep
        )
        text = _clean_text(text)
        data.append({"text": text, "label": label})
    return data


def generate_synthetic_data(
    output_path: str, num_samples: int = 10000, domain: str = "movies", seed: int = 42
) -> pd.DataFrame:
    """Generate a single-domain synthetic classification dataset."""
    random.seed(seed)
    half = num_samples // 2
    pos = _generate_batch(domain, half, label=1, is_positive=True)
    neg = _generate_batch(domain, half, label=0, is_positive=False)
    df = pd.DataFrame(pos + neg).sample(frac=1, random_state=seed).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Generated %d samples to %s", len(df), output_path)
    return df


def run_universal_generation(
    output_dir: str | Path,
    total_samples_per_domain: int = 5000,
    domains: dict[str, Any] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic dataset across multiple domains and combine them."""
    random.seed(seed)
    domains = domains or DEFAULT_DOMAINS
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for domain_key in domains:
        half = total_samples_per_domain // 2
        pos = _generate_batch(domain_key, half, label=1, is_positive=True)
        neg = _generate_batch(domain_key, half, label=0, is_positive=False)
        df = pd.DataFrame(pos + neg).sample(frac=1, random_state=seed).reset_index(drop=True)
        df.to_csv(out_dir / f"data_{domain_key}.csv", index=False)
        all_dfs.append(df)

    final_df = (
        pd.concat(all_dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    )
    final_df.to_csv(out_dir / "combined_master_dataset.csv", index=False)
    logger.info("Universal generation complete: %d total samples across %d domains.", len(final_df), len(domains))
    return final_df
