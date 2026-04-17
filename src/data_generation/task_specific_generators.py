import pandas as pd
import numpy as np
from pathlib import Path
import random
import string
import re
from typing import List, Dict, Any

# تنظیم Seed
random.seed(42)
np.random.seed(42)

# =============================================================================
# 1. VOCABULARY BASES (Reused from previous context for consistency)
# =============================================================================

BASE_VOC = {
    "pos": ["excellent", "amazing", "superb", "wonderful", "great", "perfect", "brilliant", "outstanding"],
    "neg": ["terrible", "awful", "horrible", "bad", "poor", "worst", "disappointing", "mediocre"],
    "neutral": ["okay", "fine", "average", "normal", "standard"]
}

# =============================================================================
# TASK A: SENTIMENT REVERSAL (IRONY/SARCASM)
# =============================================================================
def generate_irony_dataset(size: int) -> pd.DataFrame:
    """Generates sarcastic sentences where positive words mean negative things."""
    data = []
    templates = [
        "What a {adj_pos} experience, I literally fell asleep.",
        "Such a {adj_pos} movie, I couldn't wait to leave the theater.",
        "Absolutely {adj_pos} service, the waiter ignored me completely.",
        "The {adj_pos} product broke after one day, what a joke.",
        "Truly {adj_pos}, I regret every second of this purchase.",
        "A {adj_pos} performance, if you enjoy watching paint dry.",
        "So {adj_pos}, my wallet is now empty and I'm angry."
    ]
    
    # Labels are NEGATIVE because the sentiment is actually bad, despite positive words
    for _ in range(size):
        adj = random.choice(BASE_VOC["pos"])
        text = random.choice(templates).format(adj_pos=adj)
        data.append({"text": text, "label": 0, "task_type": "irony"})
        
    return pd.DataFrame(data)

# =============================================================================
# TASK B: LONG-RANGE DEPENDENCY (LONG CONTEXT)
# =============================================================================
def generate_long_context_dataset(size: int) -> pd.DataFrame:
    """Generates long paragraphs where the sentiment depends on the last sentence."""
    data = []
    
    for _ in range(size):
        # Generate a long neutral/negative intro
        intro_words = random.choices(["The project started with", "Initially it seemed", "At first glance", "Despite the early signs"], k=1)
        middle = random.choice(["many issues arose", "the team struggled", "the budget was tight", "errors were frequent"])
        end_positive = random.choice(["but finally we succeeded!", "yet the result was amazing!", "however, the quality was excellent!", "still, it turned out to be perfect!"])
        end_negative = random.choice(["but it failed miserably.", "and the result was terrible.", "however, it was a disaster.", "and ultimately, it was awful."])
        
        # Randomly choose positive or negative ending
        is_pos = random.random() > 0.5
        
        # Construct long sentence
        parts = [intro_words[0], middle, " ", end_positive if is_pos else end_negative]
        text = "".join(parts).capitalize() + "."
        
        # Ensure length > 50 chars
        while len(text) < 60:
            text += f" {random.choice(['furthermore', 'additionally', 'also', 'meanwhile'])} {random.choice(BASE_VOC['pos' if is_pos else 'neg'])}"
            
        label = 1 if is_pos else 0
        data.append({"text": text, "label": label, "task_type": "long_context", "length": len(text)})
        
    return pd.DataFrame(data)

# =============================================================================
# TASK C: DOMAIN SHIFT (TRAIN ON X, TEST ON Y)
# =============================================================================
def generate_domain_shift_dataset(size: int) -> pd.DataFrame:
    """Creates datasets where training domain differs from test domain."""
    domains = ["movie", "product", "finance", "health", "tech"]
    data = []
    
    for _ in range(size):
        train_dom = random.choice(domains)
        test_dom = random.choice([d for d in domains if d != train_dom])
        
        # Simple sentiment generator based on domain keywords (simplified for demo)
        if test_dom == "movie":
            text = f"The {random.choice(BASE_VOC['pos'])} acting in this film was {random.choice(BASE_VOC['pos'])}."
            label = 1
        elif test_dom == "finance":
            text = f"The {random.choice(BASE_VOC['neg'])} stock market crash caused {random.choice(BASE_VOC['neg'])} losses."
            label = 0
        elif test_dom == "product":
            text = f"This {random.choice(BASE_VOC['pos'])} gadget works perfectly."
            label = 1
        else:
            text = f"A {random.choice(BASE_VOC['pos'])} experience overall."
            label = 1
            
        # Force some negatives for balance
        if random.random() < 0.5:
            label = 0
            text = text.replace("perfectly", "poorly").replace("works", "fails")
            
        data.append({
            "text": text, 
            "label": label, 
            "train_domain": train_dom, 
            "test_domain": test_dom, 
            "task_type": "domain_shift"
        })
        
    return pd.DataFrame(data)

# =============================================================================
# TASK D: SEMANTIC PARAPHRASE EQUIVALENCE
# =============================================================================
def generate_paraphrase_dataset(size: int) -> pd.DataFrame:
    """Generates pairs of sentences with same meaning but different words."""
    data = []
    
    paraphrases = [
        ("This movie is great.", "I loved this film immensely."),
        ("The product is broken.", "It doesn't work at all."),
        ("The service was slow.", "They took forever to respond."),
        ("He is very smart.", "His intelligence is remarkable."),
        ("The food tasted bad.", "The meal was disgusting."),
        ("She runs fast.", "Her speed is incredible.")
    ]
    
    for _ in range(size // 2):
        s1, s2 = random.choice(paraphrases)
        # Determine label based on s1 (assuming s1 determines sentiment)
        pos_keywords = ["great", "loved", "immensely", "remarkable", "fast", "incredible", "perfect", "good"]
        neg_keywords = ["broken", "doesn't work", "slow", "took forever", "bad", "disgusting", "terrible"]
        
        label = 1 if any(k in s1.lower() for k in pos_keywords) else 0
        
        data.append({"text": s1, "label": label, "task_type": "paraphrase_pair_1"})
        data.append({"text": s2, "label": label, "task_type": "paraphrase_pair_2"})
        
    return pd.DataFrame(data)

# =============================================================================
# TASK E: LOW-RESOURCE NOISE SENSITIVITY
# =============================================================================
def add_heavy_noise(text: str) -> str:
    """Applies aggressive noise."""
    words = list(text)
    
    # 1. Case Randomization
    for i in range(len(words)):
        if random.random() < 0.3:
            words[i] = words[i].swapcase()
            
    # 2. Typos (Swap/Delete)
    for i in range(len(words)):
        if random.random() < 0.15 and words[i].isalpha():
            if random.random() < 0.5 and i > 0:
                words[i], words[i-1] = words[i-1], words[i]
            else:
                words.pop(i)
                break
                
    # 3. Remove Punctuation
    punct_indices = [i for i, c in enumerate(words) if c in string.punctuation]
    for idx in punct_indices[:len(punct_indices)//2]:
        words[idx] = ""
        
    # 4. Word Repetition
    if random.random() < 0.2:
        w_list = text.split()
        if len(w_list) > 1:
            idx = random.randint(0, len(w_list)-1)
            w_list.insert(idx, w_list[idx])
            return " ".join(w_list)
            
    return "".join(words)

def generate_noise_dataset(size: int) -> pd.DataFrame:
    """Generates clean sentences then adds heavy noise."""
    base_texts = [
        "The movie was excellent.", "The product is terrible.", 
        "Service was good.", "Food was bad.", "Tech is amazing.", 
        "Health is important.", "Money matters."
    ]
    
    data = []
    for _ in range(size):
        clean_text = random.choice(base_texts)
        noisy_text = add_heavy_noise(clean_text)
        
        # Label based on original clean text
        pos_words = ["excellent", "good", "amazing", "important"]
        neg_words = ["terrible", "bad"]
        label = 1 if any(w in clean_text.lower() for w in pos_words) else 0
        
        data.append({
            "text": noisy_text, 
            "label": label, 
            "task_type": "noise_sensitivity",
            "clean_original": clean_text # For debugging
        })
        
    return pd.DataFrame(data)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_task_generations(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tasks = {
        "task_a_irony": generate_irony_dataset(1000),
        "task_b_long_context": generate_long_context_dataset(1000),
        "task_c_domain_shift": generate_domain_shift_dataset(1000),
        "task_d_paraphrase": generate_paraphrase_dataset(1000),
        "task_e_noise": generate_noise_dataset(1000)
    }
    
    print("🚀 Generating Specialized Tasks...")
    
    for task_name, df in tasks.items():
        file_path = output_dir / f"{task_name}.csv"
        df.to_csv(file_path, index=False)
        print(f"✅ Generated {task_name}: {len(df)} samples")
        
    # Optional: Combine all into one master file with a 'task_type' column
    combined = pd.concat(tasks.values(), ignore_index=True)
    combined.to_csv(output_dir / "all_specialized_tasks.csv", index=False)
    print(f"\n✨ All specialized tasks saved to {output_dir}")

if __name__ == "__main__":
    run_all_task_generations(Path.cwd() / "data")