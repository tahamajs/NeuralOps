import pandas as pd
import numpy as np
from pathlib import Path
import random
import string
import re
from typing import List, Dict, Tuple, Any
import json

# ==================== CONFIGURATION ====================
random.seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("./data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== VOCABULARY BASES ====================
VOCAB = {
    "pos_adj": ["excellent", "amazing", "superb", "wonderful", "great", "perfect", 
                "brilliant", "outstanding", "fantastic", "terrific", "awesome", "exceptional",
                "phenomenal", "marvelous", "splendid", "magnificent", "impressive", "delightful"],
    "neg_adj": ["terrible", "awful", "horrible", "bad", "poor", "worst", "disappointing", 
                "mediocre", "dreadful", "abysmal", "pathetic", "lousy", "subpar", 
                "unacceptable", "inferior", "defective", "faulty", "inadequate"],
    "neutral_adj": ["okay", "fine", "average", "normal", "standard", "decent", "acceptable"],
    "pos_nouns": ["masterpiece", "triumph", "delight", "treasure", "gem", "success", "blessing"],
    "neg_nouns": ["disaster", "nightmare", "failure", "waste", "catastrophe", "mess", "disappointment"],
    "pos_verbs": ["love", "adore", "enjoy", "appreciate", "recommend", "praise", "celebrate"],
    "neg_verbs": ["hate", "despise", "regret", "avoid", "condemn", "criticize", "reject"],
    "intensifiers": ["extremely", "incredibly", "absolutely", "totally", "completely", 
                     "utterly", "highly", "remarkably", "exceptionally", "profoundly"],
    "negation_words": ["not", "never", "no", "neither", "nor", "hardly", "barely", "scarcely"],
    "comparators_better": ["better", "superior", "greater", "more impressive", "superb", "excellent"],
    "comparators_worse": ["worse", "inferior", "less impressive", "worse", "terrible", "awful"],
}

# ==================== TASK 1: IRONY/SARCASM DETECTION ====================
def generate_irony_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    تسک ۱: تشخیص طعنه و طنز
    جملاتی که کلمات مثبت دارن ولی معنی منفی می‌دن
    """
    templates_irony = [
        "What a {adj} experience, I literally fell asleep.",
        "Such a {adj} movie, I couldn't wait to leave the theater.",
        "Absolutely {adj} service, the waiter ignored me completely.",
        "The {adj} product broke after one day, what a joke.",
        "Truly {adj}, I regret every second of this purchase.",
        "A {adj} performance, if you enjoy watching paint dry.",
        "So {adj}, my wallet is now empty and I'm angry.",
        "Oh great, another {adj} day of this.",
        "Fantastic, just fantastic, now everything is {adj}.",
        "Wonderful, just {adj}, now I've to fix everything.",
        "Perfect, absolutely {adj}, this is exactly what I needed.",
        "How {adj} of you to finally respond after weeks.",
        "Thanks for the {adj} help, really appreciate it.",
        "Oh lovely, another {adj} surprise waiting for me.",
        "Great, just {adj}, nothing could be better.",
    ]
    
    # Non-ironic positive examples
    templates_positive = [
        "This is an {adj} experience that I will never forget.",
        "The movie was absolutely {adj}, highly recommend it.",
        "Such {adj} service, the staff was very helpful.",
        "The product quality is {adj}, worth every penny.",
        "What a {adj} performance by the entire cast.",
        "I had an {adj} time at this wonderful place.",
        "The experience was {adj} and very memorable.",
        "This is truly {adj}, exceeded all expectations.",
        "An {adj} result that makes everyone proud.",
        "The quality is {adj}, couldn't be happier.",
    ]
    
    data = []
    
    # Irony = Negative label (0) despite positive words
    for _ in range(n_samples // 2):
        adj = random.choice(VOCAB["pos_adj"])
        text = random.choice(templates_irony).format(adj=adj)
        data.append({"text": text, "label": 0, "task_type": "irony"})
    
    # Genuine positive = Positive label (1)
    for _ in range(n_samples // 2):
        adj = random.choice(VOCAB["pos_adj"])
        text = random.choice(templates_positive).format(adj=adj)
        data.append({"text": text, "label": 1, "task_type": "irony"})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ==================== TASK 2: NEGATION HANDLING ====================
def generate_negation_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    تسک ۲: مدیریت نقیضه
    "not bad" = positive, "not good" = negative
    """
    templates_double_neg = [
        "This is not {neg_adj} at all.",
        "It's not as {neg_adj} as people say.",
        "This is hardly {neg_adj}.",
        "It's not particularly {neg_adj}.",
        "Nothing {neg_adj} about this experience.",
        "It's not {neg_adj}, it's actually quite good.",
        "I wouldn't call it {neg_adj}.",
        "It's far from {neg_adj}.",
        "Not {neg_adj}, not perfect, but acceptable.",
        "This is not {neg_adj} by any means.",
    ]
    
    templates_negated_pos = [
        "This is not {pos_adj} at all.",
        "It's not as {pos_adj} as I expected.",
        "This is hardly {pos_adj}.",
        "It's not particularly {pos_adj}.",
        "Nothing {pos_adj} about this experience.",
        "It's not {pos_adj}, it's actually quite bad.",
        "I wouldn't call it {pos_adj}.",
        "It's far from {pos_adj}.",
        "Not {pos_adj}, more like disappointing.",
        "This is not {pos_adj} by any stretch.",
    ]
    
    templates_simple_pos = [
        "This is {pos_adj}.",
        "It's really {pos_adj}.",
        "Absolutely {pos_adj} experience.",
        "Very {pos_adj} and enjoyable.",
        "Truly {pos_adj}.",
        "An {pos_adj} result indeed.",
        "Quite {pos_adj} overall.",
        "Really {pos_adj} work.",
        "The performance was {pos_adj}.",
        "This is {pos_adj} and I'm happy.",
    ]
    
    templates_simple_neg = [
        "This is {neg_adj}.",
        "It's really {neg_adj}.",
        "Absolutely {neg_adj} experience.",
        "Very {neg_adj} and unpleasant.",
        "Truly {neg_adj}.",
        "An {neg_adj} result indeed.",
        "Quite {neg_adj} overall.",
        "Really {neg_adj} work.",
        "The performance was {neg_adj}.",
        "This is {neg_adj} and I'm disappointed.",
    ]
    
    data = []
    
    # Double negative = Positive (1)
    for _ in range(n_samples // 4):
        neg_adj = random.choice(VOCAB["neg_adj"])
        text = random.choice(templates_double_neg).format(neg_adj=neg_adj)
        data.append({"text": text, "label": 1, "task_type": "negation"})
    
    # Negated positive = Negative (0)
    for _ in range(n_samples // 4):
        pos_adj = random.choice(VOCAB["pos_adj"])
        text = random.choice(templates_negated_pos).format(pos_adj=pos_adj)
        data.append({"text": text, "label": 0, "task_type": "negation"})
    
    # Simple positive = Positive (1)
    for _ in range(n_samples // 4):
        pos_adj = random.choice(VOCAB["pos_adj"])
        text = random.choice(templates_simple_pos).format(pos_adj=pos_adj)
        data.append({"text": text, "label": 1, "task_type": "negation"})
    
    # Simple negative = Negative (0)
    for _ in range(n_samples // 4):
        neg_adj = random.choice(VOCAB["neg_adj"])
        text = random.choice(templates_simple_neg).format(neg_adj=neg_adj)
        data.append({"text": text, "label": 0, "task_type": "negation"})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ==================== TASK 3: COMPARATIVE SENTIMENT ====================
def generate_comparative_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    تسک ۳: تحلیل احساس مقایسه‌ای
    "X is better than Y" vs "X is worse than Y"
    """
    subjects = ["This product", "The movie", "This service", "The book", 
                "This restaurant", "The game", "The experience", "The performance"]
    
    objects = ["the previous version", "the competition", "others", "the original",
               "what I expected", "the standard", "the norm", "the alternative"]
    
    templates_better = [
        "{sub} is {comp} than {obj}.",
        "{sub} is far {comp} compared to {obj}.",
        "{sub} proves to be {comp} when compared with {obj}.",
        "In comparison to {obj}, {sub} is {comp}.",
        "{sub} stands out as {comp} against {obj}.",
        "Unlike {obj}, {sub} is {comp}.",
        "While {obj} is mediocre, {sub} is {comp}.",
        "{sub} definitely feels {comp} than {obj}.",
        "If you compare {sub} with {obj}, it's {comp}.",
        "The difference is clear: {sub} is {comp} to {obj}.",
    ]
    
    templates_worse = [
        "{sub} is {comp} than {obj}.",
        "{sub} is far {comp} compared to {obj}.",
        "{sub} proves to be {comp} when compared with {obj}.",
        "In comparison to {obj}, {sub} is {comp}.",
        "{sub} stands out as {comp} against {obj}.",
        "Unlike {obj}, {sub} is {comp}.",
        "While {obj} is great, {sub} is {comp}.",
        "{sub} definitely feels {comp} than {obj}.",
        "If you compare {sub} with {obj}, it's {comp}.",
        "The difference is clear: {sub} is {comp} to {obj}.",
    ]
    
    data = []
    
    # Better = Positive (1)
    for _ in range(n_samples // 2):
        sub = random.choice(subjects)
        obj = random.choice(objects)
        comp = random.choice(VOCAB["comparators_better"])
        text = random.choice(templates_better).format(sub=sub, obj=obj, comp=comp)
        data.append({"text": text, "label": 1, "task_type": "comparative"})
    
    # Worse = Negative (0)
    for _ in range(n_samples // 2):
        sub = random.choice(subjects)
        obj = random.choice(objects)
        comp = random.choice(VOCAB["comparators_worse"])
        text = random.choice(templates_worse).format(sub=sub, obj=obj, comp=comp)
        data.append({"text": text, "label": 0, "task_type": "comparative"})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ==================== TASK 4: LONG-RANGE DEPENDENCY ====================
def generate_long_context_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    تسک ۴: وابستگی بلند‌برد
    احساس نهایی جمله به آخرین بخش بستگی داره
    """
    intros = [
        "The project started with many challenges.",
        "Initially, everything seemed to go wrong.",
        "At first glance, there were many problems.",
        "The beginning was quite difficult.",
        "Early on, we faced numerous obstacles.",
        "Starting this journey was hard.",
        "The initial phase had many issues.",
        "In the beginning, nothing worked right.",
    ]
    
    middles_pos = [
        " but with perseverance we overcame them.",
        " yet we managed to turn things around.",
        " however, the team worked tirelessly.",
        " but dedication made the difference.",
        " yet we never gave up hope.",
        " but our efforts paid off eventually.",
        " however, we found solutions to all problems.",
        " but we adapted and improved.",
    ]
    
    middles_neg = [
        " and things only got worse.",
        " and the situation deteriorated further.",
        " and unfortunately it was a complete disaster.",
        " and the problems multiplied.",
        " and it was beyond repair.",
        " and the outcome was catastrophic.",
        " and nothing could save it.",
        " and it was doomed from the start.",
    ]
    
    endings_pos = [
        "In the end, the result was amazing!",
        "Ultimately, we achieved remarkable success!",
        "The final outcome exceeded all expectations!",
        "In the end, everything turned out perfect!",
        "The conclusion was truly wonderful!",
        "Finally, we reached an excellent result!",
        "The ending was absolutely brilliant!",
        "In the end, it was a complete triumph!",
    ]
    
    endings_neg = [
        "In the end, the result was disastrous!",
        "Ultimately, it was a complete failure!",
        "The final outcome was terrible!",
        "In the end, everything fell apart!",
        "The conclusion was awful!",
        "Finally, it was a total disaster!",
        "The ending was absolutely horrible!",
        "In the end, it was a nightmare!",
    ]
    
    data = []
    
    # Positive ending = Positive (1)
    for _ in range(n_samples // 2):
        intro = random.choice(intros)
        middle = random.choice(middles_pos)
        ending = random.choice(endings_pos)
        text = intro + middle + " " + ending
        data.append({"text": text, "label": 1, "task_type": "long_context"})
    
    # Negative ending = Negative (0)
    for _ in range(n_samples // 2):
        intro = random.choice(intros)
        middle = random.choice(middles_neg)
        ending = random.choice(endings_neg)
        text = intro + middle + " " + ending
        data.append({"text": text, "label": 0, "task_type": "long_context"})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ==================== TASK 5: ASPECT-BASED SENTIMENT ====================
def generate_aspect_based_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    تسک ۵: احساس مبتنی بر جنبه
    چندین جنبه مختلف در یک متن با احساسات متفاوت
    """
    # Format: (aspect, positive_phrase, negative_phrase)
    aspects = [
        ("food", "the food was delicious", "the food was terrible"),
        ("service", "the service was excellent", "the service was poor"),
        ("ambiance", "the ambiance was lovely", "the ambiance was awful"),
        ("price", "the price was reasonable", "the price was excessive"),
        ("quality", "the quality was outstanding", "the quality was subpar"),
        ("staff", "the staff was friendly", "the staff was rude"),
        ("location", "the location was convenient", "the location was inconvenient"),
        ("cleanliness", "the place was clean", "the place was dirty"),
    ]
    
    templates_mixed = [
        "While {aspect1}, {aspect2}.",
        "On the positive side, {aspect1}, but {aspect2}.",
        "{aspect1}. However, {aspect2}.",
        "I loved {aspect1}, but hated {aspect2}.",
        "{aspect1}. Unfortunately, {aspect2}.",
        "The best part was {aspect1}, though {aspect2}.",
        "{aspect1}. On the downside, {aspect2}.",
        "Great {aspect1}, but poor {aspect2}.",
    ]
    
    templates_all_pos = [
        "{aspect1} and {aspect2} were both excellent.",
        "Everything was great, especially {aspect1} and {aspect2}.",
        "I loved {aspect1} and {aspect2}.",
        "Both {aspect1} and {aspect2} exceeded expectations.",
        "The experience was wonderful with {aspect1} and {aspect2}.",
    ]
    
    templates_all_neg = [
        "{aspect1} and {aspect2} were both terrible.",
        "Everything was awful, especially {aspect1} and {aspect2}.",
        "I hated {aspect1} and {aspect2}.",
        "Both {aspect1} and {aspect2} were disappointing.",
        "The experience was horrible with {aspect1} and {aspect2}.",
    ]
    
    data = []
    
    # Mixed sentiment - majority wins
    for _ in range(n_samples // 3):
        asp1 = random.choice(aspects)
        asp2 = random.choice(aspects)
        
        # 2 pos, 1 neg = Positive
        if random.random() < 0.5:
            text = random.choice(templates_mixed).format(
                aspect1=asp1[1], 
                aspect2=asp2[2]  # negative
            )
            data.append({"text": text, "label": 1, "task_type": "aspect_based"})
        else:
            text = random.choice(templates_mixed).format(
                aspect1=asp1[2],  # negative
                aspect2=asp2[1]
            )
            data.append({"text": text, "label": 0, "task_type": "aspect_based"})
    
    # All positive = Positive (1)
    for _ in range(n_samples // 3):
        asp1 = random.choice(aspects)
        asp2 = random.choice(aspects)
        text = random.choice(templates_all_pos).format(
            aspect1=asp1[1], 
            aspect2=asp2[1]
        )
        data.append({"text": text, "label": 1, "task_type": "aspect_based"})
    
    # All negative = Negative (0)
    for _ in range(n_samples // 3):
        asp1 = random.choice(aspects)
        asp2 = random.choice(aspects)
        text = random.choice(templates_all_neg).format(
            aspect1=asp1[2], 
            aspect2=asp2[2]
        )
        data.append({"text": text, "label": 0, "task_type": "aspect_based"})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ==================== TASK 6: CONDITIONAL SENTIMENT ====================
def generate_conditional_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    تسک ۶: احساس شرطی
    احساس به شرط بستگی داره
    """
    conditions_pos = [
        "if you like action movies",
        "if you enjoy thrillers",
        "if you're a fan of comedies",
        "if you appreciate good storytelling",
        "if you want quality entertainment",
        "if you like romantic movies",
        "if you enjoy horror films",
        "if you're into sci-fi",
    ]
    
    conditions_neg = [
        "if you hate violence",
        "if you dislike slow pacing",
        "if you can't stand bad acting",
        "if you want a quick watch",
        "if you prefer serious content",
        "if you don't like romance",
        "if you're sensitive to horror",
        "if you avoid sci-fi",
    ]
    
    conclusions_pos = [
        "you'll love this.",
        "this is perfect for you.",
        "you'll really enjoy it.",
        "this will delight you.",
        "it's absolutely great.",
        "you'll find it amazing.",
        "it's exactly what you need.",
        "this is made for you.",
    ]
    
    conclusions_neg = [
        "you'll hate this.",
        "this is not for you.",
        "you'll really dislike it.",
        "it will disappoint you.",
        "it's absolutely terrible.",
        "you'll find it awful.",
        "it's not worth your time.",
        "avoid this at all costs.",
    ]
    
    templates = [
        "{condition}, {conclusion}",
        "If {condition}, {conclusion}",
        "{condition}? {conclusion}",
        "Well, {condition}, {conclusion}",
    ]
    
    data = []
    
    # Positive condition + Positive conclusion = Positive (1)
    for _ in range(n_samples // 4):
        cond = random.choice(conditions_pos)
        conc = random.choice(conclusions_pos)
        text = random.choice(templates).format(condition=cond, conclusion=conc)
        data.append({"text": text, "label": 1, "task_type": "conditional"})
    
    # Negative condition + Negative conclusion = Negative (0)
    for _ in range(n_samples // 4):
        cond = random.choice(conditions_neg)
        conc = random.choice(conclusions_neg)
        text = random.choice(templates).format(condition=cond, conclusion=conc)
        data.append({"text": text, "label": 0, "task_type": "conditional"})
    
    # Positive condition + Negative conclusion = Negative (0)
    for _ in range(n_samples // 4):
        cond = random.choice(conditions_pos)
        conc = random.choice(conclusions_neg)
        text = random.choice(templates).format(condition=cond, conclusion=conc)
        data.append({"text": text, "label": 0, "task_type": "conditional"})
    
    # Negative condition + Positive conclusion = Positive (1)
    for _ in range(n_samples // 4):
        cond = random.choice(conditions_neg)
        conc = random.choice(conclusions_pos)
        text = random.choice(templates).format(condition=cond, conclusion=conc)
        data.append({"text": text, "label": 1, "task_type": "conditional"})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ==================== TASK 7: NOISE ROBUSTNESS ====================
def generate_noise_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    تسک ۷: مقاومت در برابر نویز
    متن با تایپی، نویسه‌های اضافی و...
    """
    base_positive = [
        "The movie was excellent and I loved it.",
        "This product is amazing and works perfectly.",
        "The service was wonderful and very helpful.",
        "The food was delicious and fresh.",
        "The experience was fantastic and memorable.",
        "The quality is outstanding and reliable.",
        "The performance was brilliant and captivating.",
        "The book was engaging and insightful.",
    ]
    
    base_negative = [
        "The movie was terrible and I hated it.",
        "This product is awful and broken completely.",
        "The service was horrible and very rude.",
        "The food was disgusting and cold.",
        "The experience was disappointing and boring.",
        "The quality is poor and unreliable.",
        "The performance was boring and dull.",
        "The book was confusing and poorly written.",
    ]
    
    def add_noise(text: str) -> str:
        words = list(text)
        
        # Case randomization (30%)
        for i in range(len(words)):
            if random.random() < 0.3 and words[i].isalpha():
                words[i] = words[i].swapcase()
        
        # Character swap (typos) (15%)
        for i in range(len(words)):
            if random.random() < 0.15 and words[i].isalpha() and i > 0:
                if random.random() < 0.5:
                    words[i], words[i-1] = words[i-1], words[i]
        
        # Character deletion (10%)
        for i in range(len(words)):
            if random.random() < 0.1 and words[i].isalpha() and len(words) > 5:
                words[i] = ""
                break
        
        # Random character insertion (10%)
        if random.random() < 0.1:
            insert_pos = random.randint(0, len(words)-1)
            random_char = random.choice(string.ascii_lowercase)
            words.insert(insert_pos, random_char)
        
        # Word repetition (10%)
        if random.random() < 0.1:
            word_list = text.split()
            if len(word_list) > 2:
                idx = random.randint(0, len(word_list)-1)
                word_list.insert(idx, word_list[idx])
                return " ".join(word_list)
        
        return "".join(words)
    
    data = []
    
    # Positive text with noise
    for _ in range(n_samples // 2):
        clean = random.choice(base_positive)
        noisy = add_noise(clean)
        data.append({"text": noisy, "label": 1, "task_type": "noise_robustness", "clean_text": clean})
    
    # Negative text with noise
    for _ in range(n_samples // 2):
        clean = random.choice(base_negative)
        noisy = add_noise(clean)
        data.append({"text": noisy, "label": 0, "task_type": "noise_robustness", "clean_text": clean})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ==================== TASK 8: DOMAIN SHIFT ====================
def generate_domain_shift_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    تسک ۸: جابجایی دامنه
    یادگیری در یک دامنه، تست در دامنه دیگر
    """
    domains = {
        "movie": {
            "pos": ["the acting was superb", "the plot was compelling", "the cinematography was stunning"],
            "neg": ["the acting was terrible", "the plot was boring", "the cinematography was awful"]
        },
        "product": {
            "pos": ["the quality is excellent", "the build is solid", "the features are useful"],
            "neg": ["the quality is poor", "the build is flimsy", "the features are useless"]
        },
        "restaurant": {
            "pos": ["the food was delicious", "the service was impeccable", "the ambiance was lovely"],
            "neg": ["the food was terrible", "the service was slow", "the ambiance was noisy"]
        },
        "book": {
            "pos": ["the writing was eloquent", "the story was engaging", "the characters were deep"],
            "neg": ["the writing was poor", "the story was dull", "the characters were flat"]
        },
        "hotel": {
            "pos": ["the room was clean", "the staff was friendly", "the location was convenient"],
            "neg": ["the room was dirty", "the staff was rude", "the location was far"]
        },
    }
    
    templates = [
        "{phrase} and I highly recommend it.",
        "{phrase}, definitely worth checking out.",
        "{phrase}, a great experience overall.",
        "{phrase}, exceeded my expectations.",
        "{phrase}, truly remarkable.",
    ]
    
    data = []
    
    # Generate from one domain, test on another
    for _ in range(n_samples):
        train_domain = random.choice(list(domains.keys()))
        test_domain = random.choice([d for d in domains.keys() if d != train_domain])
        
        # Use test domain for the actual text
        is_positive = random.random() > 0.5
        phrase = random.choice(domains[test_domain]["pos" if is_positive else "neg"])
        text = random.choice(templates).format(phrase=phrase)
        
        data.append({
            "text": text, 
            "label": 1 if is_positive else 0, 
            "task_type": "domain_shift",
            "train_domain": train_domain,
            "test_domain": test_domain
        })
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ==================== TASK 9: PARAPHRASE DETECTION ====================
def generate_paraphrase_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    تسک ۹: تشخیص بازنویسی
    جملات با معنی یکسان اما کلمات متفاوت
    """
    synonym_pairs = {
        "good": ["great", "excellent", "fine", "decent", "solid", "awesome"],
        "bad": ["terrible", "awful", "poor", "dreadful", "horrible", "lousy"],
        "happy": ["glad", "pleased", "delighted", "thrilled", "joyful"],
        "sad": ["unhappy", "upset", "disappointed", "dejected", "sorrowful"],
        "love": ["adore", "cherish", "appreciate", "treasure", "value"],
        "hate": ["despise", "detest", "loathe", "abhor", "dislike"],
        "big": ["large", "huge", "enormous", "vast", "massive"],
        "small": ["tiny", "little", "miniature", "compact", "petite"],
        "fast": ["quick", "rapid", "swift", "speedy", "blazing"],
        "slow": ["sluggish", "gradual", "leisurely", "tardy", "unhurried"],
    }
    
    base_sentences_pos = [
        "This is a good movie.",
        "I had a happy experience.",
        "The service was great.",
        "I love this product.",
        "It's a big improvement.",
        "The performance was fast.",
        "This is a good result.",
        "I'm happy with this.",
    ]
    
    base_sentences_neg = [
        "This is a bad movie.",
        "I had a sad experience.",
        "The service was terrible.",
        "I hate this product.",
        "It's a big disappointment.",
        "The performance was slow.",
        "This is a bad result.",
        "I'm disappointed with this.",
    ]
    
    def paraphrase(text: str) -> str:
        words = text.lower().split()
        new_words = []
        for w in words:
            clean_w = w.strip(string.punctuation)
            if clean_w in synonym_pairs:
                rep = random.choice(synonym_pairs[clean_w])
                if w.endswith('.') or w.endswith('!'):
                    rep += w[-1]
                new_words.append(rep)
            else:
                new_words.append(w)
        return " ".join(new_words)
    
    data = []
    
    # Positive paraphrases
    for _ in range(n_samples // 2):
        base = random.choice(base_sentences_pos)
        para = paraphrase(base)
        # Add variation
        if random.random() < 0.3:
            prefix = random.choice(["Absolutely", "Definitely", "Truly", "Really"])
            para = f"{prefix}, {para.lower()}"
        data.append({"text": para, "label": 1, "task_type": "paraphrase"})
    
    # Negative paraphrases
    for _ in range(n_samples // 2):
        base = random.choice(base_sentences_neg)
        para = paraphrase(base)
        if random.random() < 0.3:
            prefix = random.choice(["Unfortunately", "Regrettably", "Sad to say"])
            para = f"{prefix}, {para.lower()}"
        data.append({"text": para, "label": 0, "task_type": "paraphrase"})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ==================== TASK 10: TEMPORAL/CONTEXTUAL SENTIMENT ====================
def generate_temporal_context_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    تسک ۱۰: احساس زمانی/موقیتی
    احساس به زمان و موقیت بستگی داره
    """
    temporal_pos = [
        "After years of waiting, it was worth it.",
        "Finally, something good happened.",
        "This made my day completely.",
        "The best thing that happened this week.",
        "After all the struggles, success feels amazing.",
        "This is exactly what I needed right now.",
        "In these difficult times, this brings hope.",
        "This made up for all the bad days.",
    ]
    
    temporal_neg = [
        "After years of waiting, it was a waste.",
        "Finally, something else went wrong.",
        "This ruined my day completely.",
        "The worst thing that happened this week.",
        "After all the struggles, failure is devastating.",
        "This is the last thing I needed right now.",
        "In these good times, this brings disappointment.",
        "This made an already bad day worse.",
    ]
    
    context_pos = [
        "Given the circumstances, this is impressive.",
        "Considering the challenges, this is remarkable.",
        "For a first attempt, this is excellent.",
        "Under the pressure, this was great.",
        "Despite the limitations, this works well.",
        "With limited resources, this is amazing.",
        "Against all odds, this succeeded.",
        "In this situation, it's actually good.",
    ]
    
    context_neg = [
        "Given the circumstances, this is disappointing.",
        "Considering the challenges, this is unacceptable.",
        "For a first attempt, this is terrible.",
        "Under the pressure, this failed badly.",
        "Despite the resources, this doesn't work.",
        "With all the resources, this is awful.",
        "With all the help, this failed.",
        "In this situation, it's actually bad.",
    ]
    
    data = []
    
    # Temporal positive = Positive (1)
    for _ in range(n_samples // 4):
        text = random.choice(temporal_pos)
        data.append({"text": text, "label": 1, "task_type": "temporal_context"})
    
    # Temporal negative = Negative (0)
    for _ in range(n_samples // 4):
        text = random.choice(temporal_neg)
        data.append({"text": text, "label": 0, "task_type": "temporal_context"})
    
    # Context positive = Positive (1)
    for _ in range(n_samples // 4):
        text = random.choice(context_pos)
        data.append({"text": text, "label": 1, "task_type": "temporal_context"})
    
    # Context negative = Negative (0)
    for _ in range(n_samples // 4):
        text = random.choice(context_neg)
        data.append({"text": text, "label": 0, "task_type": "temporal_context"})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ==================== MAIN EXECUTION ====================
def generate_all_datasets(n_samples_per_task: int = 1000) -> Dict[str, pd.DataFrame]:
    """
    تولید تمام ۱۰ دیتاست
    """
    print("=" * 70)
    print("🎯 GENERATING 10 CHALLENGING REASONING TASKS")
    print("=" * 70)
    
    tasks = {
        "01_irony": generate_irony_dataset(n_samples_per_task),
        "02_negation": generate_negation_dataset(n_samples_per_task),
        "03_comparative": generate_comparative_dataset(n_samples_per_task),
        "04_long_context": generate_long_context_dataset(n_samples_per_task),
        "05_aspect_based": generate_aspect_based_dataset(n_samples_per_task),
        "06_conditional": generate_conditional_dataset(n_samples_per_task),
        "07_noise_robustness": generate_noise_dataset(n_samples_per_task),
        "08_domain_shift": generate_domain_shift_dataset(n_samples_per_task),
        "09_paraphrase": generate_paraphrase_dataset(n_samples_per_task),
        "10_temporal_context": generate_temporal_context_dataset(n_samples_per_task),
    }
    
    return tasks

def save_datasets(tasks: Dict[str, pd.DataFrame], output_dir: Path):
    """
    ذخیره‌سازی دیتاست‌ها
    """
    all_data = []
    
    for task_name, df in tasks.items():
        # Save individual file
        file_path = output_dir / f"{task_name}.csv"
        df.to_csv(file_path, index=False)
        
        # Add to combined
        df_copy = df.copy()
        df_copy['task_name'] = task_name
        all_data.append(df_copy)
        
        # Print stats
        pos_count = df['label'].sum()
        neg_count = len(df) - pos_count
        print(f"✅ {task_name}: {len(df)} samples | Pos: {pos_count} | Neg: {neg_count}")
    
    # Save combined dataset
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_path = output_dir / "all_tasks_combined.csv"
    combined_df.to_csv(combined_path, index=False)
    
    print("-" * 70)
    print(f"📊 TOTAL: {len(combined_df)} samples")
    print(f"   Positive: {combined_df['label'].sum()} | Negative: {len(combined_df) - combined_df['label'].sum()}")
    print(f"   Saved to: {combined_path}")
    print("=" * 70)
    
    return combined_df

def create_dataset_info(tasks: Dict[str, pd.DataFrame], output_dir: Path):
    """
    ایجاد فایل توضیحات دیتاست
    """
    info = {}
    
    task_descriptions = {
        "01_irony": "تشخیص طعنه و طنز - جملاتی که کلمات مثبت دارن ولی معنی منفی می‌دن",
        "02_negation": "مدیریت نقیضه - درک 'not bad' = مثبت و 'not good' = منفی",
        "03_comparative": "تحلیل احساس مقایسه‌ای - 'better than' vs 'worse than'",
        "04_long_range_dependency": "وابستگی بلند‌برد - احساس به آخرین جمله بستگی داره",
        "05_aspect_based": "احساس مبتنی بر جنبه - چندین جنبه با احساسات متفاوت",
        "06_conditional": "احساس شرطی - احساس به شرط بستگی داره",
        "07_noise_robustness": "مقاومت در برابر نویز - متن با تایپی و اشتباهات",
        "08_domain_shift": "جابجایی دامنه - یادگیری در یک دامنه، تست در دیگری",
        "09_paraphrase": "تشخیص بازنویسی - معنی یکسان با کلمات متفاوت",
        "10_temporal_context": "احساس زمانی - احساس به زمان و موقیت بستگی داره",
    }
    
    for task_name, df in tasks.items():
        info[task_name] = {
            "description": task_descriptions.get(task_name, ""),
            "samples": len(df),
            "positive_samples": int(df['label'].sum()),
            "negative_samples": int(len(df) - df['label'].sum()),
            "avg_text_length": int(df['text'].str.len().mean()),
            "reasoning_type": get_reasoning_type(task_name),
        }
    
    info_path = output_dir / "dataset_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print(f"📋 Dataset info saved to: {info_path}")

def get_reasoning_type(task_name: str) -> str:
    """
    نوع استدلال مورد نیاز برای هر تسک
    """
    reasoning_types = {
        "01_irony": "Semantics + Pragmatics - درک معنی ضمنی",
        "02_negation": "Logic - درک منطق نقیضه",
        "03_comparative": "Comparative Reasoning - تحلیل مقایسه‌ای",
        "04_long_context": "Long-range Dependency - وابستگی بلند",
        "05_aspect_based": "Multi-aspect Analysis - تحلیل چند جنبه‌ای",
        "06_conditional": "Conditional Reasoning - استدلال شرطی",
        "07_noise_robustness": "Pattern Recognition - تشخیص الگو در نویز",
        "08_domain_shift": "Domain Adaptation - تطبیق دامنه",
        "09_paraphrase": "Semantic Equivalence - هم‌ارزش معنایی",
        "10_temporal_context": "Contextual Reasoning - استدلال موقیتی",
    }
    return reasoning_types.get(task_name, "Unknown")

# ==================== RUN ====================
if __name__ == "__main__":
    # Generate all 10 datasets with 1000 samples each
    tasks = generate_all_datasets(n_samples_per_task=1000)
    
    # Save to files
    combined_df = save_datasets(tasks, OUTPUT_DIR)
    
    # Create info file
    create_dataset_info(tasks, OUTPUT_DIR)
    
    # Print sample from each
    print("\n📝 SAMPLE FROM EACH TASK:")
    print("-" * 70)
    for task_name, df in tasks.items():
        sample = df.iloc[0]
        label_str = "✅ Positive" if sample['label'] == 1 else "❌ Negative"
        print(f"\n{task_name} [{label_str}]:")
        print(f"   {sample['text'][:100]}...")