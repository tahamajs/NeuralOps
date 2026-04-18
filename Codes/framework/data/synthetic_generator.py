import pandas as pd
from pathlib import Path
import random
import string
import re
import json

random.seed(42)

DOMAINS = {
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
    # Add more domains as needed
}

SYNONYMS_DB = {
    "good": ["great", "excellent", "fine", "decent"],
    "bad": ["terrible", "awful", "poor", "dreadful"],
    "love": ["adore", "cherish", "enjoy", "appreciate"],
    "hate": ["despise", "loathe", "detest", "dislike"]
}

TEMPLATES_POS = [
    "The {adj} {noun} was {adv} {verb}.",
    "I found the {noun} to be {adj} and {adv} {verb}."
]

TEMPLATES_NEG = [
    "The {adj} {noun} was {adv} {verb}.",
    "I found the {noun} to be {adj} and {adv} {verb}."
]

INTENSIFIERS = ["extremely", "highly", "deeply", "profoundly"]
PREP_PHRASES_POS = ["with remarkable precision", "with incredible attention to detail"]
PREP_PHRASES_NEG = ["with terrible writing", "with poor execution"]

def safe_choice(lst):
    return random.choice(lst) if lst else ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in string.punctuation:
        text += "."
    return text

def generate_batch(domain_key, size, label, templates_list, prep_phrases):
    data = []
    vocab = DOMAINS[domain_key]
    for _ in range(size):
        adj = safe_choice(vocab["pos_adj"] if label == 1 else vocab["neg_adj"])
        noun = safe_choice(vocab["nouns"])
        verb = safe_choice(vocab["verbs"])
        adv = "perfectly" if label == 1 else "terribly"
        prep = safe_choice(prep_phrases)
        template = random.choice(templates_list)
        sentence = template.format(adj=adj, noun=noun, verb=verb, adv=adv, prep_phrase=prep)
        sentence = clean_text(sentence)
        data.append({"text": sentence, "label": label})
    return data

def run_universal_generation(output_dir: Path, total_samples_per_domain: int = 5000):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for domain_key in DOMAINS.keys():
        half = total_samples_per_domain // 2
        pos_batch = generate_batch(domain_key, half, 1, TEMPLATES_POS, PREP_PHRASES_POS)
        neg_batch = generate_batch(domain_key, half, 0, TEMPLATES_NEG, PREP_PHRASES_NEG)
        df = pd.DataFrame(pos_batch + neg_batch)
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv(output_dir / f"data_{domain_key}.csv", index=False)
        all_dfs.append(df)
    final_df = pd.concat(all_dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
    final_df.to_csv(output_dir / "combined_master_dataset.csv", index=False)
    print(f"Generated {len(final_df)} samples across {len(DOMAINS)} domains.")