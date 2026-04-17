import pandas as pd
import numpy as np
from pathlib import Path
import random
import string
import re
from typing import List, Dict, Any, Tuple
from itertools import product
import math

# تنظیم Seed برای تکرارپذیری کامل
random.seed(42)
np.random.seed(42)

# =============================================================================
# 1. EXPANDED VOCABULARY DATABASE (7 DOMAINS + SPECIALIZED TERMINOLOGY)
# =============================================================================

DOMAIN_VOC = {
    "movie": {
        "adj_pos": ["masterful", "breathtaking", "captivating", "heartwarming", "brilliant", "stunning", "emotional", "riveting", "flawless", "unforgettable", "visceral", "poignant"],
        "adj_neg": ["tedious", "derivative", "lifeless", "predictable", "shallow", "painful", "forgettable", "mediocre", "confusing", "abysmal", "clichéd", "soulless"],
        "noun_pos": ["performance", "cinematography", "narrative", "script", "direction", "soundtrack", "acting", "visuals", "storytelling", "artistry", "pacing", "atmosphere"],
        "noun_neg": ["plot holes", "writing", "pacing", "dialogue", "cast", "editing", "characterization", "climax", "resolution", "concept", "tonal shifts", "logic gaps"],
        "verb_pos": ["delivered", "crafted", "elevated", "transported", "captivated", "resonated", "immersed", "transformed", "surpassed", "achieved", "evoked", "probed"],
        "verb_neg": ["failed", "collapsed", "dragged", "distracted", "bored", "disappointed", "weakened", "spoiled", "ignored", "ruined", "alienated", "muddled"]
    },
    "product": {
        "adj_pos": ["reliable", "sturdy", "value-packed", "efficient", "seamless", "innovative", "robust", "intuitive", "premium", "essential", "ergonomic", "long-lasting"],
        "adj_neg": ["fragile", "flimsy", "overpriced", "broken", "defective", "slow", "confusing", "cheap", "useless", "frustrating", "bulky", "incompatible"],
        "noun_pos": ["build quality", "battery life", "customer support", "design", "features", "durability", "performance", "packaging", "interface", "warranty", "user experience", "specs"],
        "noun_neg": ["shipping time", "plastic feel", "battery drain", "instructions", "compatibility", "assembly", "glitches", "warranty claim", "quality control", "return process", "firmware bugs", "oversized box"],
        "verb_pos": ["works perfectly", "exceeded expectations", "lasted long", "improved workflow", "saved money", "enhanced experience", "delivered value", "performed well", "met needs", "surprised me", "streamlined tasks", "optimized usage"],
        "verb_neg": ["broke quickly", "stopped working", "cost too much", "didn't fit", "crashed often", "was confusing", "hurt my wallet", "wasted time", "missed the mark", "fell apart", "delayed delivery", "lacked features"]
    },
    "tweet": {
        "adj_pos": ["amazing", "inspiring", "powerful", "groundbreaking", "united", "hopeful", "progressive", "genius", "viral", "necessary", "revolutionary", "empowering"],
        "adj_neg": ["ridiculous", "hateful", "biased", "misleading", "toxic", "divisive", "weak", "scandalous", "outrageous", "disappointing", "corrupt", "dangerous"],
        "noun_pos": ["movement", "change", "community", "truth", "solidarity", "innovation", "voice", "future", "justice", "progress", "awareness", "activism"],
        "noun_neg": ["lies", "division", "corruption", "hypocrisy", "fear", "agenda", "propaganda", "scandal", "failure", "chaos", "misinformation", "apathy"],
        "verb_pos": ["stands up for", "challenges", "empowers", "connects", "breaks barriers", "sparks hope", "drives change", "supports", "celebrates", "proves", "mobilizes", "educates"],
        "verb_neg": ["promotes hate", "spreads lies", "divides us", "ignores facts", "fuels fear", "destroys trust", "silences voices", "creates chaos", "exploits", "manipulates", "distorts", "suppresses"]
    },
    "education": {
        "adj_pos": ["comprehensive", "engaging", "clear", "structured", "insightful", "practical", "enlightening", "accessible", "rigorous", "rewarding", "dynamic", "transformative"],
        "adj_neg": ["disorganized", "outdated", "superficial", "vague", "dry", "irrelevant", "confusing", "theoretical", "poorly explained", "tedious", "static", "abstract"],
        "noun_pos": ["curriculum", "instructor", "assignments", "resources", "lectures", "feedback", "syllabus", "projects", "materials", "learning path", "workshops", "labs"],
        "noun_neg": ["grading", "deadlines", "content", "tests", "homework", "platform", "schedule", "requirements", "support", "textbook", "administrative hurdles", "lecture slides"],
        "verb_pos": ["clarifies concepts", "encourages growth", "provides tools", "stimulates thought", "prepares students", "offers value", "guides learning", "builds skills", "opens doors", "deepens understanding", "facilitates mastery", "inspires curiosity"],
        "verb_neg": ["lacks depth", "confuses students", "wastes time", "fails to explain", "is outdated", "demands too much", "offers little", "hurts grades", "discourages effort", "misleads learners", "overloads students", "undermines confidence"]
    },
    "tech_news": {
        "adj_pos": ["cutting-edge", "revolutionary", "secure", "fast", "scalable", "optimized", "intelligent", "versatile", "advanced", "seamless", "robust", "future-proof"],
        "adj_neg": ["vulnerable", "bloated", "buggy", "slow", "incompatible", "proprietary", "expensive", "restrictive", "unstable", "obsolete", "fragile", "inefficient"],
        "noun_pos": ["architecture", "security protocol", "algorithm", "update", "integration", "feature set", "cloud solution", "API", "deployment", "infrastructure", "latency", "bandwidth"],
        "noun_neg": ["backdoor", "latency", "outage", "patch", "dependency", "licensing fee", "compatibility issue", "vendor lock-in", "server crash", "data breach", "technical debt", "legacy systems"],
        "verb_pos": ["secures data", "accelerates processing", "streamlines operations", "enhances security", "optimizes cost", "scales effortlessly", "integrates smoothly", "protects users", "modernizes systems", "boosts efficiency", "automates workflows", "minimizes downtime"],
        "verb_neg": ["leaks data", "slows down", "blocks access", "increases costs", "causes downtime", "compromises safety", "limits flexibility", "breaks compatibility", "consumes resources", "threatens privacy", "exposes vulnerabilities", "hinders scalability"]
    },
    "health_medical": {
        "adj_pos": ["effective", "safe", "gentle", "potent", "therapeutic", "restorative", "holistic", "evidence-based", "non-invasive", "beneficial", "precise", "timely"],
        "adj_neg": ["harmful", "ineffective", "risky", "side-effect heavy", "slow", "aggressive", "inconclusive", "contradictory", "delayed", "negligible", "toxic", "invasive"],
        "noun_pos": ["treatment", "therapy", "diagnosis", "medication", "recovery", "symptoms", "procedure", "clinical trial", "outcome", "wellness", "care plan", "dosage"],
        "noun_neg": ["complications", "adverse effects", "misdiagnosis", "allergy", "infection", "relapse", "wait time", "cost", "insurance denial", "error", "miscommunication", "protocol failure"],
        "verb_pos": ["heals", "relieves", "prevents", "monitors", "detects", "alleviates", "restores", "improves", "supports", "guides", "preserves", "enhances"],
        "verb_neg": ["worsens", "triggers", "delays", "misses", "confuses", "harms", "ignores", "obstructs", "complicates", "endangers", "postpones", "misinterprets"]
    },
    "finance_econ": {
        "adj_pos": ["profitable", "stable", "lucrative", "transparent", "secure", "liquid", "diversified", "growth-oriented", "resilient", "audited", "compliant", "strategic"],
        "adj_neg": ["volatile", "risky", "opaque", "unstable", "debt-ridden", "illiquid", "concentrated", "loss-making", "fraudulent", "regulatory", "inflated", "speculative"],
        "noun_pos": ["portfolio", "investment", "revenue", "equity", "market share", "dividend", "yield", "asset", "liquidity", "ROI", "cash flow", "valuation"],
        "noun_neg": ["losses", "liabilities", "debt", "default", "inflation", "recession", "audit failure", "tax evasion", "market crash", "volatility", "bubble", "corruption"],
        "verb_pos": ["grows", "accumulates", "protects", "generates", "maximizes", "diversifies", "hedges", "optimizes", "invests", "returns", "stabilizes", "appreciates"],
        "verb_neg": ["shrinks", "depletes", "risks", "loses", "minimizes", "concentrates", "exposes", "underperforms", "borrows", "defaults", "crashes", "erodes"]
    }
}

INTENSIFIERS_POS = ["absolutely", "truly", "undeniably", "remarkably", "exceptionally", "highly", "profoundly", "utterly", "genuinely", "incredibly", "positively", "decidedly"]
INTENSIFIERS_NEG = ["terribly", "awfully", "disappointingly", "regrettably", "painfully", "completely", "absolutely", "utterly", "sadly", "frustratingly", "extremely", "notably"]

# =============================================================================
# 2. ADVANCED GENERATION STRATEGIES
# =============================================================================

def clean_text(text: str) -> str:
    """Standardize text format."""
    text = re.sub(r'\s+', ' ', text.strip())
    if text and text[-1] not in string.punctuation:
        text += "."
    return text.capitalize()

def generate_noise(text: str, noise_type: str = "random") -> str:
    """Apply realistic noise based on type."""
    words = list(text)
    
    if noise_type == "typo":
        # Swap adjacent characters or remove a letter
        for i in range(len(words)):
            if random.random() < 0.08 and words[i].isalpha():
                if random.random() < 0.5 and i > 0:
                    words[i], words[i-1] = words[i-1], words[i]
                else:
                    words.pop(i)
                    break
    
    elif noise_type == "case_mess":
        # Random uppercase/lowercase
        for i in range(len(words)):
            if random.random() < 0.1:
                words[i] = words[i].swapcase()
    
    elif noise_type == "missing_punct":
        # Remove some punctuation
        punct_indices = [i for i, c in enumerate(words) if c in string.punctuation]
        for idx in punct_indices[:len(punct_indices)//2]:
            words[idx] = ""
            
    elif noise_type == "repeat_word":
        # Repeat a word 1-2 times
        words_list = text.split()
        if len(words_list) > 2:
            idx = random.randint(1, len(words_list)-2)
            words_list.insert(idx, words_list[idx])
            text = " ".join(words_list)
            return clean_text(text)

    return "".join(words)

def generate_complex_sentence(domain: str, label: int, complexity: str) -> str:
    voc = DOMAIN_VOC[domain]
    is_pos = (label == 1)
    
    adj_list = voc["adj_pos"] if is_pos else voc["adj_neg"]
    noun_list = voc["noun_pos"] if is_pos else voc["noun_neg"]
    verb_list = voc["verb_pos"] if is_pos else voc["verb_neg"]
    intensifiers = INTENSIFIERS_POS if is_pos else INTENSIFIERS_NEG
    
    # Complex Templates
    templates = []
    
    if complexity == "simple":
        templates = [
            f"The {random.choice(noun_list)} was {random.choice(adj_list)}.",
            f"I found the {random.choice(noun_list)} to be {random.choice(adj_list)}.",
            f"This {random.choice(noun_list)} {random.choice(verb_list)} significantly."
        ]
    elif complexity == "compound":
        templates = [
            f"The {random.choice(noun_list)} was {random.choice(adj_list)}, but the {random.choice(noun_list)} {random.choice(verb_list)} poorly.",
            f"While the {random.choice(noun_list)} was {random.choice(adj_list)}, the overall {random.choice(noun_list)} felt {random.choice(adj_list)}.",
            f"Not only did the {random.choice(noun_list)} {random.choice(verb_list)}, but it also made the {random.choice(noun_list)} {random.choice(adj_list)}."
        ]
    elif complexity == "complex_nested":
        templates = [
            f"Despite the {random.choice(adj_list)} nature of the {random.choice(noun_list)}, the {random.choice(noun_list)} {random.choice(verb_list)} in a way that was {random.choice(adj_list)}.",
            f"It is rare to see a {random.choice(noun_list)} that {random.choice(verb_list)} so {random.choice(adj_list)}, yet the {random.choice(noun_list)} remained {random.choice(adj_list)}.",
            f"Although the {random.choice(noun_list)} {random.choice(verb_list)} with {random.choice(intensifiers)} precision, the {random.choice(noun_list)} was ultimately {random.choice(adj_list)}."
        ]
    else:
        templates = [templates[0]]

    template = random.choice(templates)
    
    # Fill placeholders
    sentence = template.format(
        adj=random.choice(adj_list),
        noun=random.choice(noun_list),
        verb=random.choice(verb_list),
        intensifier=random.choice(intensifiers)
    )
    
    return clean_text(sentence)

def generate_contextual_prefix(domain: str, label: int) -> str:
    prefixes_pos = [
        "I was genuinely impressed by",
        "It is rare to find such",
        "This product/service truly delivers",
        "One of the best experiences I've had regarding",
        "The way this handled things was",
        "From a professional standpoint,"
    ]
    prefixes_neg = [
        "I was thoroughly disappointed by",
        "It is hard to ignore how",
        "This product/service failed to deliver",
        "One of the worst experiences I've had regarding",
        "The way this handled things was",
        "From a critical perspective,"
    ]
    
    prefix = random.choice(prefixes_pos if label == 1 else prefixes_neg)
    body = generate_complex_sentence(domain, label, "simple")
    return f"{prefix} {body.lower()}."

def generate_paraphrased_base(base_text: str, label: int) -> str:
    """Create variations of a base sentiment without changing meaning."""
    if label == 1:
        variations = [
            f"Truly, {base_text}",
            f"It was {base_text}",
            f"I must say, {base_text}",
            f"Absolutely, {base_text}",
            f"In my opinion, {base_text}"
        ]
    else:
        variations = [
            f"Regrettably, {base_text}",
            f"Unfortunately, {base_text}",
            f"To be honest, {base_text}",
            f"Frankly, {base_text}",
            f"Quite frankly, {base_text}"
        ]
    return clean_text(random.choice(variations))

def generate_contrastive(domain: str, label: int) -> str:
    """Generate sentences with contrast (e.g., "Good X but bad Y")."""
    voc = DOMAIN_VOC[domain]
    is_pos = (label == 1)
    
    # We need mixed sentiments here
    pos_adj = voc["adj_pos"] if is_pos else voc["adj_neg"]
    neg_adj = voc["adj_neg"] if is_pos else voc["adj_pos"]
    pos_verb = voc["verb_pos"] if is_pos else voc["verb_neg"]
    neg_verb = voc["verb_neg"] if is_pos else voc["verb_pos"]
    nouns = voc["noun_pos"] if is_pos else voc["noun_neg"]
    
    # Pick two different nouns from the same category (one positive context, one negative context)
    n1 = random.choice(nouns)
    n2 = random.choice(nouns)
    while n1 == n2:
        n2 = random.choice(nouns)
        
    templates = [
        f"The {n1} was {random.choice(pos_adj)}, yet the {n2} was {random.choice(neg_adj)}.",
        f"While the {n1} {random.choice(pos_verb)} beautifully, the {n2} {random.choice(neg_verb)} terribly.",
        f"Despite the {random.choice(pos_adj)} {n1}, the {n2} turned out to be {random.choice(neg_adj)}."
    ]
    
    return clean_text(random.choice(templates))

# =============================================================================
# 3. MAIN GENERATION PIPELINE (COMPREHENSIVE)
# =============================================================================

def run_comprehensive_generation(output_dir: Path, total_samples: int = 10000):
    """
    Generates a highly diverse synthetic dataset across multiple domains, 
    complexities, and noise levels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    domains = list(DOMAIN_VOC.keys())
    complexities = ["simple", "compound", "complex_nested"]
    noise_types = ["none", "typo", "case_mess", "missing_punct", "repeat_word"]
    
    # Calculate distribution
    samples_per_domain = total_samples // len(domains)
    half_pos = samples_per_domain // 2
    
    all_data = []
    
    print(f"🚀 Starting COMPREHENSIVE Generation... Target: {total_samples} samples")
    print("-" * 60)
    
    for domain in domains:
        domain_data = []
        
        # Generate for each complexity level
        for comp in complexities:
            count = half_pos
            
            # Positive samples
            for _ in range(count):
                strategy = random.choice(["direct", "contextual", "paraphrase", "contrastive"])
                
                if strategy == "direct":
                    text = generate_complex_sentence(domain, 1, comp)
                elif strategy == "contextual":
                    text = generate_contextual_prefix(domain, 1)
                elif strategy == "paraphrase":
                    base = generate_complex_sentence(domain, 1, "simple")
                    text = generate_paraphrased_base(base, 1)
                else: # contrastive
                    text = generate_contrastive(domain, 1)
                
                # Apply noise randomly (30% chance)
                if random.random() < 0.3:
                    noise = random.choice(noise_types)
                    text = generate_noise(text, noise)
                
                domain_data.append({"text": text, "label": 1, "domain": domain, "complexity": comp, "noise": "yes" if random.random() < 0.3 else "no"})
            
            # Negative samples
            for _ in range(count):
                strategy = random.choice(["direct", "contextual", "paraphrase", "contrastive"])
                
                if strategy == "direct":
                    text = generate_complex_sentence(domain, 0, comp)
                elif strategy == "contextual":
                    text = generate_contextual_prefix(domain, 0)
                elif strategy == "paraphrase":
                    base = generate_complex_sentence(domain, 0, "simple")
                    text = generate_paraphrased_base(base, 0)
                else: # contrastive
                    text = generate_contrastive(domain, 0)
                
                # Apply noise randomly
                if random.random() < 0.3:
                    noise = random.choice(noise_types)
                    text = generate_noise(text, noise)
                
                domain_data.append({"text": text, "label": 0, "domain": domain, "complexity": comp, "noise": "yes" if random.random() < 0.3 else "no"})
        
        df_domain = pd.DataFrame(domain_data)
        df_domain = df_domain.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save individual domain file
        df_domain.to_csv(output_dir / f"data_{domain}.csv", index=False)
        all_data.append(df_domain)
        
        print(f"✅ Domain '{domain}': Generated {len(df_domain)} samples (Pos: {df_domain['label'].sum()}, Neg: {len(df_domain) - df_domain['label'].sum()})")

    # Combine all datasets
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Final Shuffle
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save Master File
    master_path = output_dir / "combined_master_dataset.csv"
    final_df.to_csv(master_path, index=False)
    
    # Print Summary Statistics
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY - ULTRA-COMPREHENSIVE SYNTHETIC DATASET")
    print("="*60)
    print(f"Total Samples:     {len(final_df)}")
    print(f"Positive (1):      {final_df['label'].sum()} ({final_df['label'].mean()*100:.1f}%)")
    print(f"Negative (0):      {len(final_df) - final_df['label'].sum()} ({(1-final_df['label'].mean())*100:.1f}%)")
    print(f"Domains Covered:   {', '.join(domains)}")
    print(f"Complexities:      {', '.join(complexities)}")
    print(f"Noise Applied:     ~30% of samples")
    print(f"Master File:       {master_path}")
    print("-" * 60)
    print("📝 Sample Texts (Mixed Domains, Complexities & Noise):")
    
    # Show a diverse sample
    samples_to_show = min(10, len(final_df))
    for i, row in final_df.head(samples_to_show).iterrows():
        status = "✅ POS" if row['label'] == 1 else "❌ NEG"
        dom = row['domain'].upper()
        comp = row['complexity']
        noise_status = "🔇 NOISE" if row['noise'] == 'yes' else "🔇 CLEAN"
        preview = row['text'][:70] + "..." if len(row['text']) > 70 else row['text']
        print(f"  [{status}] {dom:12} | {comp:15} | {noise_status:10} | {preview}")
    print("="*60)
    
    return final_df

if __name__ == "__main__":
    # Execute generation
    df_generated = run_comprehensive_generation(Path.cwd() / "data", total_samples=10000)
    
    # Optional: Verify file exists
    master_file = Path.cwd() / "data" / "combined_master_dataset.csv"
    if master_file.exists():
        print(f"\n✨ SUCCESS! Dataset ready at: {master_file}")
        print("This dataset is now robust enough for rigorous SVD optimization testing.")
    else:
        print("\n⚠️ Error generating file.")