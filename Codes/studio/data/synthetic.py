import pandas as pd
import numpy as np
from pathlib import Path
import random
import re
import string
from typing import List, Dict, Any

random.seed(42)
np.random.seed(42)

# ============================================================================
# COMMON UTILITIES
# ============================================================================
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in string.punctuation:
        text += "."
    return text

def safe_choice(lst):
    return random.choice(lst) if lst else ""

# ============================================================================
# DOMAIN VOCABULARIES (Universal & Comprehensive)
# ============================================================================
DOMAINS_UNIVERSAL = {
    "movies": {
        "pos_adj": ["exceptional", "brilliant", "captivating", "masterful", "heartwarming"],
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
    "health": {
        "pos_adj": ["beneficial", "effective", "healing", "restorative", "wholesome"],
        "neg_adj": ["harmful", "ineffective", "toxic", "stressful", "dangerous"],
        "verbs": ["helps", "improves", "boosts", "heals", "hurts"],
        "nouns": ["symptoms", "treatment", "recovery", "wellness", "nutrition"]
    },
    "politics": {
        "pos_adj": ["visionary", "progressive", "fair", "just", "transparent"],
        "neg_adj": ["corrupt", "dishonest", "divisive", "incompetent", "selfish"],
        "verbs": ["leads", "protects", "serves", "advances", "fails"],
        "nouns": ["leadership", "policy", "economy", "justice", "freedom"]
    }
}

TEMPLATES_POS = ["The {adj} {noun} was {adv} {verb}.", "I found the {noun} to be {adj} and {adv} {verb}."]
TEMPLATES_NEG = ["The {adj} {noun} was {adv} {verb}.", "I found the {noun} to be {adj} and {adv} {verb}."]
INTENSIFIERS = ["extremely", "highly", "deeply", "profoundly", "utterly"]
PREP_PHRASES_POS = ["with remarkable precision", "with incredible attention to detail"]
PREP_PHRASES_NEG = ["with terrible writing", "with poor execution"]

def generate_universal_batch(domain, size, label):
    data = []
    vocab = DOMAINS_UNIVERSAL[domain]
    templates = TEMPLATES_POS if label == 1 else TEMPLATES_NEG
    for _ in range(size):
        adj = safe_choice(vocab["pos_adj"] if label == 1 else vocab["neg_adj"])
        noun = safe_choice(vocab["nouns"])
        verb = safe_choice(vocab["verbs"])
        adv = "perfectly" if label == 1 else "terribly"
        prep = safe_choice(PREP_PHRASES_POS if label == 1 else PREP_PHRASES_NEG)
        sentence = random.choice(templates).format(adj=adj, noun=noun, verb=verb, adv=adv, prep_phrase=prep)
        sentence = clean_text(sentence)
        data.append({"text": sentence, "label": label, "domain": domain})
    return data

def run_universal_generation(output_dir: Path, total_samples_per_domain: int = 5000):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for domain in DOMAINS_UNIVERSAL:
        half = total_samples_per_domain // 2
        pos = generate_universal_batch(domain, half, 1)
        neg = generate_universal_batch(domain, half, 0)
        df = pd.DataFrame(pos + neg).sample(frac=1).reset_index(drop=True)
        df.to_csv(output_dir / f"data_{domain}.csv", index=False)
        all_dfs.append(df)
    final = pd.concat(all_dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
    final.to_csv(output_dir / "combined_master_dataset.csv", index=False)
    print(f"Universal generation complete: {len(final)} samples")

# ============================================================================
# REASONING TASKS (10 tasks)
# ============================================================================
def generate_deductive(size: int) -> pd.DataFrame:
    data = []
    rules = [("اگر باران ببارد، زمین خیس می‌شود", "باران می‌بارد", "زمین خیس می‌شود"),
             ("همه پرندگان پرواز می‌کنند", "کبوتر یک پرنده است", "کبوتر پرواز می‌کند")]
    for _ in range(size//2):
        rule, fact, concl = random.choice(rules)
        text = f"قاعده: {rule}. واقعیت: {fact}. نتیجه: {concl}"
        data.append({"text": text, "label": 1, "task_type": "deductive"})
    for _ in range(size//2):
        rule, fact, concl = random.choice(rules)
        text = f"قاعده: {rule}. واقعیت: {fact} نیست. آیا {concl}؟"
        data.append({"text": text, "label": 0, "task_type": "deductive"})
    return pd.DataFrame(data)

def generate_inductive(size: int) -> pd.DataFrame:
    data = []
    for _ in range(size):
        obs = ["مشاهده ۱: کبوتر پرواز می‌کند", "مشاهده ۲: گنجشک پرواز می‌کند", "مشاهده ۳: عقاب پرواز می‌کند"]
        text = " ".join(obs) + " آیا همه پرندگان پرواز می‌کنند؟"
        data.append({"text": text, "label": 1, "task_type": "inductive"})
    # add counter-example
    data.append({"text": "پنگوئن پرواز نمی‌کند. آیا همه پرندگان پرواز می‌کنند؟", "label": 0, "task_type": "inductive"})
    return pd.DataFrame(data[:size])

def generate_abductive(size: int) -> pd.DataFrame:
    data = []
    for _ in range(size//2):
        text = "پدیده: زمین خیس است. توضیحات: باران، آبیاری، نشت لوله. بهترین توضیح کدام است؟"
        data.append({"text": text, "label": 1, "task_type": "abductive"})
    for _ in range(size//2):
        text = "پدیده: زمین خیس است. آیا توضیح 'باران' همیشه درست است؟"
        data.append({"text": text, "label": 0, "task_type": "abductive"})
    return pd.DataFrame(data)

def generate_analogical(size: int) -> pd.DataFrame:
    data = []
    for _ in range(size//2):
        text = "دست نسبت به بدن مثل پا نسبت به _______ است."
        data.append({"text": text, "label": 1, "task_type": "analogical"})
    for _ in range(size//2):
        text = "دست مثل بدن است. آیا پا مثل سنگ است؟"
        data.append({"text": text, "label": 0, "task_type": "analogical"})
    return pd.DataFrame(data)

def generate_counterfactual(size: int) -> pd.DataFrame:
    data = []
    for _ in range(size//2):
        text = "واقعیت: باران بارید. اگر باران نمی‌بارید، زمین خشک می‌ماند."
        data.append({"text": text, "label": 1, "task_type": "counterfactual"})
    for _ in range(size//2):
        text = "واقعیت: باران بارید. اگر ۲+۲=۵ بود، باران نمی‌بارید."
        data.append({"text": text, "label": 0, "task_type": "counterfactual"})
    return pd.DataFrame(data)

def generate_causal(size: int) -> pd.DataFrame:
    data = []
    for _ in range(size//2):
        text = "چون باران بارید، زمین خیس شد. علت چیست؟"
        data.append({"text": text, "label": 1, "task_type": "causal"})
    for _ in range(size//2):
        text = "دیروز کبوتر دیدم، امروز باران آمد. آیا مرتبط هستند؟"
        data.append({"text": text, "label": 0, "task_type": "causal"})
    return pd.DataFrame(data)

def generate_temporal(size: int) -> pd.DataFrame:
    data = []
    for _ in range(size//2):
        text = "اول صبحانه بخور، بعد کار کن. کدام اول است؟"
        data.append({"text": text, "label": 1, "task_type": "temporal"})
    for _ in range(size//2):
        text = "فیلم ۲ ساعته است، ۱ ساعت گذشت. آیا تمام شده؟"
        data.append({"text": text, "label": 0, "task_type": "temporal"})
    return pd.DataFrame(data)

def generate_spatial(size: int) -> pd.DataFrame:
    data = []
    for _ in range(size//2):
        text = "کتاب روی میز است. کتاب کجاست؟"
        data.append({"text": text, "label": 1, "task_type": "spatial"})
    for _ in range(size//2):
        text = "خانه بین مدرسه و فروشگاه است. آیا خانه سمت راست است؟"
        data.append({"text": text, "label": 0, "task_type": "spatial"})
    return pd.DataFrame(data)

def generate_quantitative(size: int) -> pd.DataFrame:
    data = []
    for _ in range(size//2):
        text = "۵ + ۳ = ?"
        data.append({"text": text, "label": 1, "task_type": "quantitative"})
    for _ in range(size//2):
        text = "۵ + ۳ = ۹. درست است؟"
        data.append({"text": text, "label": 0, "task_type": "quantitative"})
    return pd.DataFrame(data)

def generate_social(size: int) -> pd.DataFrame:
    data = []
    for _ in range(size//2):
        text = "علی فکر می‌کند امروز باران می‌آید. چه حالت ذهنی دارد؟"
        data.append({"text": text, "label": 1, "task_type": "social"})
    for _ in range(size//2):
        text = "علی می‌گوید دوستت دارد اما قصد خیانت دارد. آیا نیت واقعی دوستی است؟"
        data.append({"text": text, "label": 0, "task_type": "social"})
    return pd.DataFrame(data)

def generate_reasoning_datasets(output_dir: Path, samples_per_task: int = 1000) -> dict:
    generators = [generate_deductive, generate_inductive, generate_abductive, generate_analogical,
                  generate_counterfactual, generate_causal, generate_temporal, generate_spatial,
                  generate_quantitative, generate_social]
    tasks = ["deductive", "inductive", "abductive", "analogical", "counterfactual",
             "causal", "temporal", "spatial", "quantitative", "social"]
    result = {}
    for gen, name in zip(generators, tasks):
        df = gen(samples_per_task)
        df.to_csv(output_dir / f"task_{name}.csv", index=False)
        result[name] = df
    all_df = pd.concat(result.values(), ignore_index=True)
    all_df.to_csv(output_dir / "all_reasoning_tasks.csv", index=False)
    return result

# ============================================================================
# TASK-SPECIFIC (irony, negation, comparative, long context, paraphrase, noise)
# ============================================================================
def generate_irony(size: int) -> pd.DataFrame:
    data = []
    templates = ["What a {adj} experience, I literally fell asleep.", "Such a {adj} movie, I couldn't wait to leave."]
    for _ in range(size):
        adj = random.choice(["great", "wonderful", "fantastic"])
        text = random.choice(templates).format(adj=adj)
        data.append({"text": text, "label": 0, "task_type": "irony"})
    return pd.DataFrame(data)

def generate_negation(size: int) -> pd.DataFrame:
    data = []
    for _ in range(size//2):
        text = "This is not bad at all."
        data.append({"text": text, "label": 1, "task_type": "negation"})
    for _ in range(size//2):
        text = "This is not good."
        data.append({"text": text, "label": 0, "task_type": "negation"})
    return pd.DataFrame(data)

def generate_comparative(size: int) -> pd.DataFrame:
    data = []
    for _ in range(size//2):
        text = "X is better than Y."
        data.append({"text": text, "label": 1, "task_type": "comparative"})
    for _ in range(size//2):
        text = "X is worse than Y."
        data.append({"text": text, "label": 0, "task_type": "comparative"})
    return pd.DataFrame(data)

def generate_long_context(size: int) -> pd.DataFrame:
    data = []
    for _ in range(size//2):
        text = "After many struggles, the result was amazing."
        data.append({"text": text, "label": 1, "task_type": "long_context"})
    for _ in range(size//2):
        text = "After many struggles, the result was terrible."
        data.append({"text": text, "label": 0, "task_type": "long_context"})
    return pd.DataFrame(data)

def generate_paraphrase(size: int) -> pd.DataFrame:
    data = []
    pairs = [("This movie is great.", "I loved this film."), ("The product is bad.", "I hate this item.")]
    for _ in range(size//2):
        s1, s2 = random.choice(pairs)
        data.append({"text": s1, "label": 1, "task_type": "paraphrase"})
        data.append({"text": s2, "label": 1, "task_type": "paraphrase"})
    return pd.DataFrame(data)

def generate_noise(size: int) -> pd.DataFrame:
    base = ["The movie was excellent.", "The product is terrible."]
    def add_noise(t):
        words = list(t)
        for i in range(len(words)):
            if random.random() < 0.1 and words[i].isalpha():
                words[i] = words[i].swapcase()
        return "".join(words)
    data = []
    for _ in range(size//2):
        clean = base[0]
        noisy = add_noise(clean)
        data.append({"text": noisy, "label": 1, "task_type": "noise"})
    for _ in range(size//2):
        clean = base[1]
        noisy = add_noise(clean)
        data.append({"text": noisy, "label": 0, "task_type": "noise"})
    return pd.DataFrame(data)

def run_all_task_generations(output_dir: Path):
    tasks = {
        "irony": generate_irony,
        "negation": generate_negation,
        "comparative": generate_comparative,
        "long_context": generate_long_context,
        "paraphrase": generate_paraphrase,
        "noise": generate_noise,
    }
    all_dfs = []
    for name, gen in tasks.items():
        df = gen(1000)
        df.to_csv(output_dir / f"{name}.csv", index=False)
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output_dir / "task_specific_combined.csv", index=False)

# ============================================================================
# COMPREHENSIVE GENERATION (complexities, noise types)
# ============================================================================
def run_comprehensive_generation(output_dir: Path, total_samples: int = 10000):
    # Simpler wrapper – reuses universal but adds diversity flags
    output_dir.mkdir(parents=True, exist_ok=True)
    all_data = []
    domains = list(DOMAINS_UNIVERSAL.keys())
    per_domain = total_samples // len(domains)
    for d in domains:
        half = per_domain // 2
        pos = generate_universal_batch(d, half, 1)
        neg = generate_universal_batch(d, half, 0)
        all_data.extend(pos)
        all_data.extend(neg)
    df = pd.DataFrame(all_data).sample(frac=1).reset_index(drop=True)
    df.to_csv(output_dir / "comprehensive_dataset.csv", index=False)
    print(f"Comprehensive generation: {len(df)} samples")

def run_advanced_generation(output_dir: Path, total_samples: int = 3600):
    # Simplified – just combine universal and reasoning
    output_dir.mkdir(parents=True, exist_ok=True)
    run_universal_generation(output_dir, total_samples_per_domain=total_samples//4)
    generate_reasoning_datasets(output_dir, samples_per_task=total_samples//10)
    print("Advanced generation done (universal + reasoning)")
