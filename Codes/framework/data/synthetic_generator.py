# framework/data/synthetic_generator.py
"""
Synthetic data generation for text classification.
"""

import random
import re
import string
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# Default domains for synthetic data generation
DEFAULT_DOMAINS = {
    "movies": {
        "pos_adj": ["exceptional", "phenomenal", "brilliant", "masterful", "captivating"],
        "neg_adj": ["disastrous", "tedious", "bland", "lifeless", "confusing"],
        "verbs": ["delivered", "crafted", "executed", "failed", "collapsed"],
        "nouns": ["narrative", "performance", "cinematography", "script", "direction"]
    },
    "tech": {
        "pos_adj": ["incredible", "seamless", "intuitive", "powerful", "lightning-fast"],
        "neg_adj": ["buggy", "slow", "unreliable", "clunky", "overpriced"],
        "verbs": ["works", "performs", "loads", "crashes", "freezes"],
        "nouns": ["interface", "performance", "battery life", "screen quality", "processing speed"]
    },
    "restaurants": {
        "pos_adj": ["delicious", "amazing", "flavorful", "fresh", "authentic"],
        "neg_adj": ["terrible", "bland", "overpriced", "greasy", "stale"],
        "verbs": ["served", "prepared", "cooked", "presented", "plated"],
        "nouns": ["food", "service", "atmosphere", "presentation", "value"]
    },
    "books": {
        "pos_adj": ["engaging", "thought-provoking", "well-written", "captivating", "insightful"],
        "neg_adj": ["boring", "poorly-written", "tedious", "confusing", "disappointing"],
        "verbs": ["writes", "describes", "narrates", "develops", "concludes"],
        "nouns": ["plot", "character", "writing style", "pacing", "dialogue"]
    }
}


TEMPLATES_POS = [
    "The {adj} {noun} was {adv} {verb}.",
    "I found the {noun} to be {adj} and {adv} {verb}.",
    "The {adj} {noun} {prep_phrase}.",
    "What an {adj} {noun}! It {verb} {adv}."
]

TEMPLATES_NEG = [
    "The {adj} {noun} was {adv} {verb}.",
    "I found the {noun} to be {adj} and {adv} {verb}.",
    "Unfortunately, the {adj} {noun} {prep_phrase}.",
    "What a {adj} {noun}! It {verb} {adv}."
]

INTENSIFIERS = ["extremely", "highly", "deeply", "profoundly", "remarkably"]
PREP_PHRASES_POS = ["with remarkable precision", "with incredible attention to detail", "in every aspect"]
PREP_PHRASES_NEG = ["with terrible execution", "with poor craftsmanship", "in many ways"]


def safe_choice(lst):
    """Safely choose random element from list."""
    return random.choice(lst) if lst else ""


def clean_text(text: str) -> str:
    """Clean and format generated text."""
    text = re.sub(r'\s+', ' ', text).strip()
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in string.punctuation:
        text += "."
    return text


def generate_batch(
    domain_key: str,
    size: int,
    label: int,
    templates_list: List[str],
    prep_phrases: List[str],
    intensifiers: Optional[List[str]] = None
) -> List[Dict]:
    """Generate a batch of synthetic samples."""
    data = []
    vocab = DEFAULT_DOMAINS.get(domain_key, DEFAULT_DOMAINS["movies"])
    intensifiers = intensifiers or INTENSIFIERS
    
    for _ in range(size):
        if label == 1:
            adj = safe_choice(vocab["pos_adj"])
        else:
            adj = safe_choice(vocab["neg_adj"])
        
        noun = safe_choice(vocab["nouns"])
        verb = safe_choice(vocab["verbs"])
        adv = "perfectly" if label == 1 else "terribly"
        prep = safe_choice(prep_phrases)
        intensifier = safe_choice(intensifiers) if random.random() > 0.7 else ""
        
        template = random.choice(templates_list)
        
        # Randomly add intensifier
        if intensifier and random.random() > 0.5:
            adj = f"{intensifier} {adj}"
        
        sentence = template.format(
            adj=adj,
            noun=noun,
            verb=verb,
            adv=adv,
            prep_phrase=prep
        )
        sentence = clean_text(sentence)
        data.append({"text": sentence, "label": label})
    
    return data


def run_universal_generation(
    output_dir: Path,
    total_samples_per_domain: int = 5000,
    domains: Optional[Dict] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic dataset across multiple domains.
    
    Args:
        output_dir: Output directory for generated files
        total_samples_per_domain: Number of samples per domain
        domains: Custom domain vocabulary (uses default if None)
        seed: Random seed for reproducibility
        
    Returns:
        Combined DataFrame with all generated samples
    """
    random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    domains = domains or DEFAULT_DOMAINS
    all_dfs = []
    
    print(f"🔨 Generating synthetic data for {len(domains)} domains...")
    
    for domain_key in domains.keys():
        half = total_samples_per_domain // 2
        
        pos_batch = generate_batch(
            domain_key, half, 1, TEMPLATES_POS, PREP_PHRASES_POS
        )
        neg_batch = generate_batch(
            domain_key, half, 0, TEMPLATES_NEG, PREP_PHRASES_NEG
        )
        
        df = pd.DataFrame(pos_batch + neg_batch)
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Save per-domain file
        df.to_csv(output_dir / f"data_{domain_key}.csv", index=False)
        all_dfs.append(df)
        
        print(f"   ✅ Generated {len(df)} samples for '{domain_key}'")
    
    # Combine all domains
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    
    # Save combined dataset
    combined_path = output_dir / "combined_master_dataset.csv"
    final_df.to_csv(combined_path, index=False)
    
    print(f"\n✅ Generated {len(final_df)} total samples")
    print(f"   Saved to: {combined_path}")
    
    # Print class distribution
    pos_count = (final_df['label'] == 1).sum()
    neg_count = (final_df['label'] == 0).sum()
    print(f"   Class distribution: Positive: {pos_count}, Negative: {neg_count}")
    
    return final_df


def generate_synthetic_data(
    output_path: str,
    num_samples: int = 10000,
    domain: str = "movies",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate simple synthetic dataset for a single domain.
    
    Args:
        output_path: Path to save the generated CSV
        num_samples: Number of samples to generate
        domain: Domain to use ("movies", "tech", "restaurants", "books")
        seed: Random seed
        
    Returns:
        Generated DataFrame
    """
    random.seed(seed)
    
    half = num_samples // 2
    
    pos_batch = generate_batch(domain, half, 1, TEMPLATES_POS, PREP_PHRASES_POS)
    neg_batch = generate_batch(domain, half, 0, TEMPLATES_NEG, PREP_PHRASES_NEG)
    
    df = pd.DataFrame(pos_batch + neg_batch)
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {len(df)} samples to {output_path}")
    
    return df