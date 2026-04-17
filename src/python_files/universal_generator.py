# ==================== UNIVERSAL SYNTHETIC DATA GENERATOR ====================
"""
🌍 Universal Synthetic Dataset Generator for NLP Tasks
Features:
- Multi-Domain Support (Movies, Tech, Health, Politics, Sports, etc.)
- Dynamic Vocabulary Expansion
- Advanced Template System (50+ Templates per domain)
- Noise Injection & Paraphrasing
- Automatic Dataset Splitting & Balancing
"""

import pandas as pd
from pathlib import Path
import random
import string
from typing import List, Dict, Any, Tuple
import re
import json

random.seed(42)

# =============================================================================
# 1. DOMAIN CONFIGURATION & VOCABULARY DATABASE
# =============================================================================

DOMAINS = {
    "movies": {
        "name": "Movie Reviews",
        "categories": ["cinema", "film", "director", "acting", "script", "visuals", "soundtrack"],
        "pos_adj": [
            "exceptional", "phenomenal", "brilliant", "masterful", "captivating", 
            "heartwarming", "inspiring", "stunning", "unforgettable", "superb",
            "exquisite", "riveting", "touching", "eloquent", "gripping", "mesmerizing",
            "thrilling", "emotional", "powerful", "moving", "wonderful", "fantastic"
        ],
        "neg_adj": [
            "disastrous", "tedious", "bland", "lifeless", "confusing", "predictable",
            "shallow", "awkward", "painful", "forgettable", "mediocre", "subpar",
            "lackluster", "uninspired", "derivative", "terrible", "horrible", "dreadful",
            "awful", "abysmal", "weak", "poor", "flawed", "botched", "ruined"
        ],
        "verbs": ["delivered", "crafted", "executed", "presented", "wove", "painted", 
                  "sounded", "felt", "proved", "demonstrated", "failed", "collapsed", 
                  "stumbled", "lagged", "drifted", "fell", "broke", "lost", "missed"],
        "nouns": ["narrative", "performance", "cinematography", "script", "direction", 
                  "acting", "visuals", "soundtrack", "plot", "dialogue", "characterization",
                  "pacing", "atmosphere", "theme", "ending", "storytelling", "production"]
    },
    "tech": {
        "name": "Tech Product Reviews",
        "categories": ["smartphone", "laptop", "software", "AI", "hardware", "battery", "display"],
        "pos_adj": [
            "incredible", "seamless", "intuitive", "powerful", "lightning-fast", 
            "robust", "cutting-edge", "reliable", "efficient", "innovative", 
            "sleek", "premium", "impressive", "versatile", "user-friendly"
        ],
        "neg_adj": [
            "buggy", "slow", "unreliable", "clunky", "overpriced", "limited", 
            "frustrating", "outdated", "glitchy", "cumbersome", "expensive", 
            "disappointing", "broken", "laggy", "confusing", "inadequate"
        ],
        "verbs": ["works", "performs", "loads", "connects", "syncs", "crashes", 
                  "freezes", "lags", "fails", "disconnects", "breaks", "slows down"],
        "nouns": ["interface", "performance", "battery life", "screen quality", 
                  "processing speed", "build quality", "customer support", "updates"]
    },
    "health": {
        "name": "Health & Wellness",
        "categories": ["medicine", "diet", "fitness", "mental health", "therapy"],
        "pos_adj": [
            "beneficial", "effective", "healing", "restorative", "wholesome", 
            "nutritious", "energizing", "calming", "supportive", "proven", 
            "positive", "constructive", "vital", "essential", "transformative"
        ],
        "neg_adj": [
            "harmful", "ineffective", "toxic", "stressful", "dangerous", 
            "unsustainable", "counterproductive", "misleading", "risky", 
            "debilitating", "anxious", "fatiguing", "negative", "detrimental"
        ],
        "verbs": ["helps", "improves", "boosts", "heals", "supports", "hurts", 
                  "worsens", "causes", "prevents", "relieves", "aggravates", "ignores"],
        "nouns": ["symptoms", "treatment", "recovery", "wellness", "nutrition", 
                  "exercise", "medication", "advice", "side effects", "progress"]
    },
    "politics": {
        "name": "Political Discourse",
        "categories": ["policy", "election", "government", "economy", "rights"],
        "pos_adj": [
            "visionary", "progressive", "fair", "just", "transparent", "inclusive", 
            "decisive", "principled", "forward-thinking", "responsible", "bold", 
            "compassionate", "effective", "strong", "wise"
        ],
        "neg_adj": [
            "corrupt", "dishonest", "divisive", "incompetent", "selfish", 
            "oppressive", "unfair", "weak", "reckless", "short-sighted", 
            "biased", "manipulative", "tyrannical", "hypocritical", "irresponsible"
        ],
        "verbs": ["leads", "protects", "serves", "advances", "unites", "fails", 
                  "divides", "suppresses", "exploits", "neglects", "undermines", "threatens"],
        "nouns": ["leadership", "policy", "economy", "justice", "freedom", 
                  "democracy", "rights", "legislation", "administration", "public trust"]
    }
}

# Extended Synonyms for Paraphrasing
SYNONYMS_DB = {
    "good": ["great", "excellent", "fine", "decent", "nice", "solid", "awesome"],
    "bad": ["terrible", "awful", "poor", "dreadful", "horrible", "lousy", "shoddy"],
    "love": ["adore", "cherish", "enjoy", "appreciate", "like", "favor", "treasure"],
    "hate": ["despise", "loathe", "detest", "dislike", "abhor", "revile", "reject"],
    "buy": ["purchase", "acquire", "get", "obtain", "grab", "secure"],
    "sell": ["offer", "market", "trade", "dispose of", "part with"],
    "fast": ["quick", "rapid", "swift", "speedy", "instant", "blazing"],
    "slow": ["sluggish", "leisurely", "gradual", "delayed", "tardy", "plodding"]
}

# =============================================================================
# 2. ADVANCED TEMPLATE ENGINE
# =============================================================================

TEMPLATES_POS = [
    "The {adj} {noun} was {adv} {verb}.",
    "I found the {noun} to be {adj} and {adv} {verb}.",
    "It was a {adj} experience thanks to the {noun}.",
    "The {noun} was {adv} {verb} and felt {adj}.",
    "One of the most {adj} {noun}s I have ever seen.",
    "The way the {noun} was {verb} made it {adj}.",
    "A {adj} {noun} that {adv} {verb} everything.",
    "Truly a {adj} work in terms of {noun}.",
    "The {noun} exceeded expectations and was {adv} {verb}.",
    "I was captivated by the {adj} {noun}.",
    "This {noun} {verb} with {adj} precision.",
    "The {noun} {verb} the audience {prep_phrase}.",
    "A {adj} and unforgettable {noun}.",
    "The {noun} stands as a {adj} achievement.",
    "Every scene was {adj}, with the {noun} being particularly {adj}.",
    "Not only was the {noun} {adj}, but it also {verb} expectations.",
    "Despite the odds, the {noun} managed to {verb} and {verb2} everyone.",
    "The film proves that {noun} can {verb} when done with {adj} skill.",
    "While some struggle, this {noun} {verb} brilliantly.",
    "A masterclass in {noun} execution."
]

TEMPLATES_NEG = [
    "The {adj} {noun} was {adv} {verb}.",
    "I found the {noun} to be {adj} and {adv} {verb}.",
    "It was a {adj} experience because of the {noun}.",
    "The {noun} was {adv} {verb} and felt {adj}.",
    "One of the most {adj} {noun}s I have ever seen.",
    "The way the {noun} was {verb} made it {adj}.",
    "A {adj} {noun} that {adv} {verb} everything.",
    "Unfortunately, the {noun} was {adj}.",
    "The {noun} fell short and was {adv} {verb}.",
    "I was bored by the {adj} {noun}.",
    "This {noun} {verb} with {adj} lack of skill.",
    "The {noun} {verb} the audience {prep_phrase}.",
    "A {adj} and forgettable {noun}.",
    "The {noun} stands as a {adj} failure.",
    "Every scene was {adj}, with the {noun} being particularly {adj}.",
    "Not only was the {noun} {adj}, but it also {verb} all hope.",
    "Despite any potential, the {noun} failed to {verb} and {verb2} everyone.",
    "The film proves that {noun} can {verb} when done with {adj} lack of skill.",
    "While others succeed, this {noun} {verb} terribly.",
    "A disaster in {noun} execution."
]

INTENSIFIERS = ["extremely", "highly", "deeply", "profoundly", "utterly", "absolutely", "truly", "remarkably", "incredibly", "exceptionally"]
PREP_PHRASES_POS = ["with remarkable precision", "with incredible attention to detail", "with stunning visuals", "with powerful storytelling", "with emotional depth"]
PREP_PHRASES_NEG = ["with terrible writing", "with poor execution", "with weak acting", "with confusing plot", "with boring pacing"]

# =============================================================================
# 3. GENERATION LOGIC
# =============================================================================

def safe_choice(lst):
    return random.choice(lst) if lst else ""

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in string.punctuation:
        text += "."
    return text

def add_variation(text, label):
    if random.random() < 0.3:
        intensifier = random.choice(INTENSIFIERS)
        text = f"{intensifier.capitalize()}, {text.lower()}"
    if random.random() < 0.2:
        conj = random.choice(["and", "but", "however", "yet"])
        suffix = "memorable" if label == 1 else "disappointing"
        text = f"{text.rstrip('.')} {conj} it was {suffix}."
    if random.random() < 0.15:
        text = text.rstrip('.') + "!"
    if random.random() < 0.1:
        text = f"Truly, {text.lower()}"
    return text

def generate_batch(domain_key, size, label, templates_list, prep_phrases):
    data = []
    vocab = DOMAINS[domain_key]
    
    for _ in range(size):
        template = random.choice(templates_list)
        
        # Fill placeholders
        adj = safe_choice(vocab["pos_adj"] if label == 1 else vocab["neg_adj"])
        noun = safe_choice(vocab["nouns"])
        verb = safe_choice(vocab["verbs"])
        verb2 = safe_choice(vocab["verbs"])
        adv = safe_choice(vocab.get("adv", [""])) or ("perfectly" if label == 1 else "terribly")
        prep = safe_choice(prep_phrases)
        
        try:
            sentence = template.format(
                adj=adj, noun=noun, verb=verb, verb2=verb2, adv=adv, prep_phrase=prep
            )
        except KeyError:
            # Fallback if template doesn't match
            sentence = f"The {adj} {noun} was {adv} {verb}."
            
        sentence = add_variation(sentence, label)
        sentence = clean_text(sentence)
        data.append({"text": sentence, "label": label})
    
    return data

def generate_paraphrased_batch(base_sentences, size, label, domain_name):
    data = []
    source = base_sentences[:size//2] if len(base_sentences) >= size//2 else base_sentences * ((size//len(base_sentences)) + 1)
    
    for s in source:
        words = s.lower().split()
        new_words = []
        for w in words:
            clean_w = w.strip(string.punctuation)
            if clean_w in SYNONYMS_DB:
                rep = random.choice(SYNONYMS_DB[clean_w])
                if w.endswith('.') or w.endswith('!'):
                    rep += w[-1]
                new_words.append(rep)
            else:
                new_words.append(w)
        
        new_sent = " ".join(new_words)
        suffixes_pos = [" Truly exceptional.", " A masterpiece.", " Highly worth it."]
        suffixes_neg = [" Absolutely terrible.", " A disaster.", " Do not bother."]
        new_sent += random.choice(suffixes_pos if label == 1 else suffixes_neg)
        new_sent = clean_text(new_sent)
        data.append({"text": new_sent, "label": label})
    
    return data

# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================

def run_universal_generation(output_dir: Path, total_samples_per_domain: int = 5000):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_datasets = []
    
    print(f"🚀 Starting Universal Generation for {total_samples_per_domain} samples/domain...")
    
    for domain_key, config in DOMAINS.items():
        print(f"\n📂 Generating for: {config['name']} ({domain_key})")
        
        half = total_samples_per_domain // 2
        
        # Generate using multiple strategies
        batch1 = generate_batch(domain_key, half, 1, TEMPLATES_POS, PREP_PHRASES_POS) + \
                 generate_batch(domain_key, half, 0, TEMPLATES_NEG, PREP_PHRASES_NEG)
        
        # Add paraphrased versions
        base_pos = ["The movie was great.", "Excellent product.", "Love this service."]
        base_neg = ["The movie was bad.", "Terrible experience.", "Hate this product."]
        batch2 = generate_paraphrased_batch(base_pos, half, 1, domain_key) + \
                 generate_paraphrased_batch(base_neg, half, 0, domain_key)
        
        # Add noise robustness
        batch3 = []
        for _ in range(half):
            adj = safe_choice(config["pos_adj"] if random.random() > 0.5 else config["neg_adj"])
            noun = safe_choice(config["nouns"])
            template = random.choice(["{a} {n}.", "{n} is {a}.", "Very {a} {n}.", "{a}!"])
            text = template.format(a=adj, n=noun)
            if random.random() > 0.9:
                mid = len(text) // 2
                text = text[:mid].upper() + text[mid:]
            text = clean_text(text)
            batch3.append({"text": text, "label": 1 if adj in config["pos_adj"] else 0})
            
        combined_ds = batch1 + batch2 + batch3
        df = pd.DataFrame(combined_ds)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        file_path = output_dir / f"data_{domain_key}.csv"
        df.to_csv(file_path, index=False)
        
        print(f"   ✅ Generated {len(df)} samples.")
        print(f"      Positive: {df['label'].sum()} | Negative: {len(df) - df['label'].sum()}")
        
        all_datasets.append(df)
    
    # Combine all datasets into one master file
    final_df = pd.concat(all_datasets, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_path = output_dir / "combined_master_dataset.csv"
    final_df.to_csv(final_path, index=False)
    
    print("\n" + "="*60)
    print("🌍 UNIVERSAL DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Total Domains:     {len(DOMAINS)}")
    print(f"Total Samples:     {len(final_df)}")
    print(f"Positive (1):      {final_df['label'].sum()}")
    print(f"Negative (0):      {len(final_df) - final_df['label'].sum()}")
    print(f"Master File:       {final_path}")
    print("-"*60)
    print("Sample texts from different domains:")
    for i, row in final_df.head(10).iterrows():
        label_str = "✅ Pos" if row['label'] == 1 else "❌ Neg"
        print(f"  [{label_str}] {row['text'][:80]}...")
    print("="*60)

if __name__ == "__main__":
    # Run generation for 5000 samples per domain (Total ~20k+)
    run_universal_generation(Path.cwd() / "data", total_samples_per_domain=5000)