"""
Synthetic text generation for multiple domains (movies, tech, restaurants, books).
"""

import random
import re
import string
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

DEFAULT_DOMAINS = {
    "movies": {
        "pos_adj": [
            "exceptional",
            "phenomenal",
            "brilliant",
            "masterful",
            "captivating",
        ],
        "neg_adj": ["disastrous", "tedious", "bland", "lifeless", "confusing"],
        "verbs": ["delivered", "crafted", "executed", "failed", "collapsed"],
        "nouns": ["narrative", "performance", "cinematography", "script", "direction"],
    },
    "tech": {
        "pos_adj": [
            "incredible",
            "seamless",
            "intuitive",
            "powerful",
            "lightning-fast",
        ],
        "neg_adj": ["buggy", "slow", "unreliable", "clunky", "overpriced"],
        "verbs": ["works", "performs", "loads", "crashes", "freezes"],
        "nouns": [
            "interface",
            "performance",
            "battery life",
            "screen quality",
            "processing speed",
        ],
    },
    "restaurants": {
        "pos_adj": ["delicious", "amazing", "flavorful", "fresh", "authentic"],
        "neg_adj": ["terrible", "bland", "overpriced", "greasy", "stale"],
        "verbs": ["served", "prepared", "cooked", "presented", "plated"],
        "nouns": ["food", "service", "atmosphere", "presentation", "value"],
    },
    "books": {
        "pos_adj": [
            "engaging",
            "thought-provoking",
            "well-written",
            "captivating",
            "insightful",
        ],
        "neg_adj": [
            "boring",
            "poorly-written",
            "tedious",
            "confusing",
            "disappointing",
        ],
        "verbs": ["writes", "describes", "narrates", "develops", "concludes"],
        "nouns": ["plot", "character", "writing style", "pacing", "dialogue"],
    },
}

TEMPLATES_POS = [
    "The {adj} {noun} was {adv} {verb}.",
    "I found the {noun} to be {adj} and {adv} {verb}.",
    "The {adj} {noun} {prep_phrase}.",
    "What an {adj} {noun}! It {verb} {adv}.",
]

TEMPLATES_NEG = [
    "The {adj} {noun} was {adv} {verb}.",
    "I found the {noun} to be {adj} and {adv} {verb}.",
    "Unfortunately, the {adj} {noun} {prep_phrase}.",
    "What a {adj} {noun}! It {verb} {adv}.",
]


def _clean_text(text: str) -> str:
    """ clean text."""
    text = re.sub(r"\s+", " ", text).strip()
    text = text[0].upper() + text[1:] if text else text
    if text and text[-1] not in string.punctuation:
        text += "."
    return text


def _generate_batch(domain_key: str, size: int, label: int, is_positive: bool) -> list:
    """ generate batch."""
    vocab = DEFAULT_DOMAINS.get(domain_key, DEFAULT_DOMAINS["movies"])
    templates = TEMPLATES_POS if is_positive else TEMPLATES_NEG
    prep_phrases = (
        [
            "with remarkable precision",
            "with incredible attention to detail",
            "in every aspect",
        ]
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
    """Generate a single‑domain synthetic classification dataset."""
    random.seed(seed)
    half = num_samples // 2
    pos = _generate_batch(domain, half, label=1, is_positive=True)
    neg = _generate_batch(domain, half, label=0, is_positive=False)
    df = pd.DataFrame(pos + neg).sample(frac=1).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} samples to {output_path}")
    return df


def run_universal_generation(
    output_dir: Path,
    total_samples_per_domain: int = 5000,
    domains: Optional[Dict] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic dataset across multiple domains and combine them."""
    random.seed(seed)
    domains = domains or DEFAULT_DOMAINS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for domain_key in domains:
        half = total_samples_per_domain // 2
        pos = _generate_batch(domain_key, half, label=1, is_positive=True)
        neg = _generate_batch(domain_key, half, label=0, is_positive=False)
        df = pd.DataFrame(pos + neg).sample(frac=1).reset_index(drop=True)
        df.to_csv(output_dir / f"data_{domain_key}.csv", index=False)
        all_dfs.append(df)
        print(f"Generated {len(df)} samples for '{domain_key}'")

    final_df = (
        pd.concat(all_dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
    )
    final_df.to_csv(output_dir / "combined_master_dataset.csv", index=False)
    print(f"Total: {len(final_df)} samples")
    return final_df
