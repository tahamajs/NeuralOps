# studio/data/synthetic.py
import random
import re
import string
import pandas as pd
from pathlib import Path

DOMAINS = {
    "movies": {
        "pos_adj": ["exceptional", "brilliant", "captivating", "masterful"],
        "neg_adj": ["disastrous", "tedious", "bland", "lifeless"],
        "verbs": ["delivered", "crafted", "executed", "failed"],
        "nouns": ["narrative", "performance", "cinematography", "script"]
    },
    "tech": {
        "pos_adj": ["incredible", "seamless", "intuitive", "powerful"],
        "neg_adj": ["buggy", "slow", "unreliable", "clunky"],
        "verbs": ["works", "performs", "loads", "crashes"],
        "nouns": ["interface", "performance", "battery", "screen"]
    }
}
TEMPLATES_POS = ["The {adj} {noun} was {adv} {verb}."]
TEMPLATES_NEG = ["The {adj} {noun} was {adv} {verb}."]
INTENSIFIERS = ["extremely", "highly"]

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = text[0].upper() + text[1:] if text else ""
    if text and text[-1] not in string.punctuation: text += "."
    return text

def generate_batch(domain, size, label, templates, prep_phrases):
    data = []
    vocab = DOMAINS[domain]
    for _ in range(size):
        adj = random.choice(vocab["pos_adj"] if label == 1 else vocab["neg_adj"])
        noun = random.choice(vocab["nouns"])
        verb = random.choice(vocab["verbs"])
        adv = "perfectly" if label == 1 else "terribly"
        prep = random.choice(prep_phrases) if prep_phrases else ""
        template = random.choice(templates)
        sent = template.format(adj=adj, noun=noun, verb=verb, adv=adv, prep_phrase=prep)
        sent = clean_text(sent)
        data.append({"text": sent, "label": label})
    return data

def run_universal_generation(output_dir: Path, total_samples_per_domain: int = 5000):
    """Generate multi-domain synthetic dataset (original 10_sample.py style)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for domain in DOMAINS:
        half = total_samples_per_domain // 2
        pos = generate_batch(domain, half, 1, TEMPLATES_POS, ["with precision"])
        neg = generate_batch(domain, half, 0, TEMPLATES_NEG, ["with errors"])
        df = pd.DataFrame(pos + neg).sample(frac=1).reset_index(drop=True)
        df.to_csv(output_dir / f"data_{domain}.csv", index=False)
        all_dfs.append(df)
    final = pd.concat(all_dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
    final.to_csv(output_dir / "combined_master_dataset.csv", index=False)
    print(f"Generated {len(final)} samples total.")

def generate_synthetic_data(output_path: str, num_samples: int = 10000, domain: str = "movies"):
    half = num_samples // 2
    pos = generate_batch(domain, half, 1, TEMPLATES_POS, [])
    neg = generate_batch(domain, half, 0, TEMPLATES_NEG, [])
    df = pd.DataFrame(pos + neg).sample(frac=1).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df