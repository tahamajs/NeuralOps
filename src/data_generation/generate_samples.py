import pandas as pd
from pathlib import Path
import random
import string
from typing import List, Dict, Any, Tuple
import re

random.seed(42)

# ==================== EXPANDED VOCABULARY ====================

POS_VOC = {
    "adj": [
        # Strong positive
        "exceptional", "phenomenal", "brilliant", "magnificent", "captivating", 
        "heartwarming", "inspiring", "masterful", "stunning", "unforgettable",
        "superb", "exquisite", "riveting", "touching", "eloquent", "gripping",
        "mesmerizing", "thrilling", "emotional", "powerful", "moving",
        # Moderate positive
        "wonderful", "fantastic", "excellent", "outstanding", "remarkable",
        "impressive", "delightful", "enchanting", "satisfying", "rewarding",
        "engaging", "compelling", "absorbing", "fascinating", "intriguing"
    ],
    "noun": [
        # Film elements
        "narrative", "performance", "cinematography", "script", "direction", 
        "acting", "visuals", "soundtrack", "plot", "dialogue", 
        "characterization", "pacing", "atmosphere", "theme", "ending",
        # Additional
        "storytelling", "cinematography", "production", "editing", "casting",
        "score", "choreography", "set design", "costume design", "visual effects",
        "screenplay", "cinematic experience", "artistry", "craftsmanship", "vision"
    ],
    "verb": [
        # Active verbs
        "delivered", "crafted", "executed", "presented", "wove", 
        "painted", "sounded", "felt", "seemed", "proved",
        "demonstrated", "offered", "provided", "achieved", "surpassed",
        # Additional
        "captivated", "enchanted", "transported", "immersed", "engaged",
        "resonated", "struck", "moved", "touched", "impressed",
        "elevated", "transformed", "brought to life", "brought out the best in"
    ],
    "adv": [
        "flawlessly", "perfectly", "brilliantly", "beautifully", "emotionally",
        "deeply", "undeniably", "remarkably", "incredibly", "absolutely",
        "truly", "genuinely", "outstandingly", "magnificently", "superbly",
        "masterfully", "expertly", "skillfully", "artistically", "tastefully"
    ],
    "prep_phrase": [
        "with remarkable precision", "with incredible attention to detail",
        "with stunning visuals", "with powerful storytelling",
        "with emotional depth", "with masterful execution",
        "with brilliant performances", "with captivating narrative",
        "with immersive cinematography", "with unforgettable moments"
    ]
}

NEG_VOC = {
    "adj": [
        # Strong negative
        "disastrous", "tedious", "bland", "lifeless", "confusing", 
        "predictable", "shallow", "awkward", "painful", "forgettable",
        "mediocre", "subpar", "lackluster", "uninspired", "derivative",
        # Additional
        "terrible", "horrible", "dreadful", "awful", "abysmal",
        "weak", "poor", "flawed", "botched", "ruined",
        "boring", "monotonous", "repetitive", "unoriginal", "clichéd"
    ],
    "noun": [
        # Film elements
        "storyline", "execution", "writing", "cast", "editing", 
        "score", "structure", "flow", "tone", "logic", 
        "motivation", "climax", "resolution", "dialogue", "concept",
        # Additional
        "acting", "performance", "direction", "cinematography", "visuals",
        "plot holes", "character development", "pacing issues", "script",
        "production value", "special effects", "budget allocation", "narrative"
    ],
    "verb": [
        # Active verbs
        "failed", "collapsed", "stumbled", "lagged", "drifted", 
        "fell", "broke", "lost", "missed", "ignored",
        "neglected", "ruined", "spoiled", "weakened", "diminished",
        # Additional
        "disappointed", "frustrated", "bored", "annoyed", "irritated",
        "wasted", "squandered", "botched", "mishandled", "fumbled"
    ],
    "adv": [
        "terribly", "awfully", "poorly", "horribly", "painfully",
        "boringly", "predictably", "pointlessly", "uselessly", "completely",
        "utterly", "absolutely", "disappointingly", "dreadfully", "miserably",
        "sadly", "unfortunately", "regrettably", " shamefully", "embarrassingly"
    ],
    "prep_phrase": [
        "with terrible writing", "with poor execution", "with weak acting",
        "with confusing plot", "with boring pacing", "with flat dialogue",
        "with obvious flaws", "with missed opportunities", "with lack of vision",
        "with no direction", "with shallow characters", "with weak script"
    ]
}

# ==================== CONTEXT TEMPLATES ====================

CONTEXT_PREFIXES_POS = [
    "I was genuinely moved by",
    "The film left me with a sense of",
    "It is rare to find such",
    "A truly remarkable experience where the",
    "One of the most satisfying movies I've seen in years because the",
    "This masterpiece delivers",
    "From start to finish, the film provides",
    "What makes this film special is",
    "The director has created",
    "This film stands out because"
]

CONTEXT_PREFIXES_NEG = [
    "I was thoroughly disappointed by",
    "The film failed to deliver any sense of",
    "It is hard to ignore how",
    "A frustrating experience where the",
    "One of the most disappointing movies recently because the",
    "This disaster of a film offers",
    "From start to finish, the film suffers from",
    "What makes this film terrible is",
    "The director has ruined",
    "This film fails because"
]

INTENSIFIERS = [
    "extremely", "highly", "deeply", "profoundly", "utterly",
    "absolutely", "truly", "remarkably", "incredibly", "exceptionally",
    "particularly", "especially", "notably", "significantly", "substantially"
]

# ==================== SENTENCE TEMPLATES ====================

SENTENCE_TEMPLATES_POS = [
    # Template 1: The {adj} {noun} was {adv} {verb}.
    "The {adj} {noun} was {adv} {verb}.",
    # Template 2: I found the {noun} to be {adj} and {adv} {verb}.
    "I found the {noun} to be {adj} and {adv} {verb}.",
    # Template 3: It was a {adj} experience thanks to the {noun}.
    "It was a {adj} experience thanks to the {noun}.",
    # Template 4: The {noun} was {adv} {verb} and felt {adj}.
    "The {noun} was {adv} {verb} and felt {adj}.",
    # Template 5: One of the most {adj} {noun}s I have ever seen.
    "One of the most {adj} {noun}s I have ever seen.",
    # Template 6: The way the {noun} was {verb} made it {adj}.
    "The way the {noun} was {verb} made it {adj}.",
    # Template 7: A {adj} {noun} that {adv} {verb} everything.
    "A {adj} {noun} that {adv} {verb} everything.",
    # Template 8: Truly a {adj} work of art in terms of {noun}.
    "Truly a {adj} work of art in terms of {noun}.",
    # Template 9: The {noun} exceeded expectations and was {adv} {verb}.
    "The {noun} exceeded expectations and was {adv} {verb}.",
    # Template 10: I was captivated by the {adj} {noun}.
    "I was captivated by the {adj} {noun}.",
    # Template 11: This {noun} {verb} with {adj} precision.
    "This {noun} {verb} with {adj} precision.",
    # Template 12: The {noun} {verb} the audience {prep_phrase}.
    "The {noun} {verb} the audience {prep_phrase}.",
    # Template 13: A {adj} and unforgettable {noun}.
    "A {adj} and unforgettable {noun}.",
    # Template 14: The {noun} stands as a {adj} achievement in cinema.
    "The {noun} stands as a {adj} achievement in cinema.",
    # Template 15: Every scene was {adj}, with the {noun} being particularly {adj}.
    "Every scene was {adj}, with the {noun} being particularly {adj}.",
]

SENTENCE_TEMPLATES_NEG = [
    # Template 1: The {adj} {noun} was {adv} {verb}.
    "The {adj} {noun} was {adv} {verb}.",
    # Template 2: I found the {noun} to be {adj} and {adv} {verb}.
    "I found the {noun} to be {adj} and {adv} {verb}.",
    # Template 3: It was a {adj} experience because of the {noun}.
    "It was a {adj} experience because of the {noun}.",
    # Template 4: The {noun} was {adv} {verb} and felt {adj}.
    "The {noun} was {adv} {verb} and felt {adj}.",
    # Template 5: One of the most {adj} {noun}s I have ever seen.
    "One of the most {adj} {noun}s I have ever seen.",
    # Template 6: The way the {noun} was {verb} made it {adj}.
    "The way the {noun} was {verb} made it {adj}.",
    # Template 7: A {adj} {noun} that {adv} {verb} everything.
    "A {adj} {noun} that {adv} {verb} everything.",
    # Template 8: Unfortunately, the {noun} was {adj}.
    "Unfortunately, the {noun} was {adj}.",
    # Template 9: The {noun} fell short and was {adv} {verb}.
    "The {noun} fell short and was {adv} {verb}.",
    # Template 10: I was bored by the {adj} {noun}.
    "I was bored by the {adj} {noun}.",
    # Template 11: This {noun} {verb} with {adj} lack of skill.
    "This {noun} {verb} with {adj} lack of skill.",
    # Template 12: The {noun} {verb} the audience {prep_phrase}.
    "The {noun} {verb} the audience {prep_phrase}.",
    # Template 13: A {adj} and forgettable {noun}.
    "A {adj} and forgettable {noun}.",
    # Template 14: The {noun} stands as a {adj} failure in cinema.
    "The {noun} stands as a {adj} failure in cinema.",
    # Template 15: Every scene was {adj}, with the {noun} being particularly {adj}.
    "Every scene was {adj}, with the {noun} being particularly {adj}.",
]

# ==================== PARAPHRASE SYNONYMS ====================

SYNONYMS = {
    "great": ["excellent", "superb", "fantastic", "wonderful", "amazing", "outstanding", "remarkable"],
    "bad": ["terrible", "awful", "horrible", "dreadful", "poor", "mediocre", "subpar"],
    "good": ["fine", "nice", "decent", "okay", "acceptable", "satisfying", "pleasant"],
    "love": ["adore", "cherish", "appreciate", "enjoy", "like", "value", "treasure"],
    "hate": ["despise", "loathe", "detest", "dislike", "abhor", "detest", "regret"],
    "amazing": ["incredible", "astonishing", "astounding", "phenomenal", "extraordinary"],
    "terrible": ["horrible", "dreadful", "appalling", "atrocious", "abysmal"],
    "beautiful": ["stunning", "gorgeous", "magnificent", "breathtaking", "exquisite"],
    "boring": ["tedious", "monotonous", "dull", "uninteresting", "tiresome"],
    "best": ["finest", "greatest", "most excellent", "top", "premier"],
    "worst": ["most terrible", "most awful", "least acceptable", "bottom", "poorest"],
    "enjoy": ["appreciate", "relish", "savor", "delight in", "love"],
    "disappoint": ["let down", "frustrate", "dissatisfy", "fail", "underwhelm"],
    "recommend": ["suggest", "advise", "endorse", "encourage", "urge"],
    "avoid": ["skip", "steer clear of", "stay away from", "refrain from", "eschew"],
    "watch": ["see", "view", "catch", "screen", "take in"],
    "film": ["movie", "cinema", "picture", "production", "motion picture"],
    "story": ["narrative", "plot", "storyline", "tale", "account"],
    "acting": ["performance", "portrayal", "execution", "delivery", "work"],
}

# ==================== BASE SENTENCES ====================

BASE_SENTENCES_POS = [
    "The movie was great.",
    "I loved the acting.",
    "The story was good.",
    "Best film ever.",
    "Highly recommended.",
    "A masterpiece of cinema.",
    "The direction was superb.",
    "Incredible performances all around.",
    "A must-watch for everyone.",
    "This film exceeded all expectations.",
    "The cinematography was breathtaking.",
    "An emotional and powerful journey.",
    "The script was brilliantly written.",
    "Characters that stay with you.",
    "A triumph of filmmaking."
]

BASE_SENTENCES_NEG = [
    "The movie was bad.",
    "I hated the acting.",
    "The story was bad.",
    "Worst film ever.",
    "Do not watch.",
    "A disaster of a film.",
    "The direction was terrible.",
    "Poor performances throughout.",
    "A complete waste of time.",
    "This film failed on every level.",
    "The cinematography was dull.",
    "A boring and pointless journey.",
    "The script was poorly written.",
    "Characters I couldn't care less about.",
    "A failure of filmmaking."
]

# ==================== CONTEXT BASES ====================

CONTEXT_BASES_POS = [
    "the performance was incredible",
    "the script was well-written",
    "the visuals were stunning",
    "the ending was satisfying",
    "the characters were relatable",
    "the storytelling was captivating",
    "the pacing was perfect",
    "the music was hauntingly beautiful",
    "the emotions were raw and real",
    "the director's vision was clear"
]

CONTEXT_BASES_NEG = [
    "the performance was weak",
    "the script was confusing",
    "the visuals were dull",
    "the ending was disappointing",
    "the characters were flat",
    "the storytelling was muddled",
    "the pacing was off",
    "the music was forgettable",
    "the emotions felt forced",
    "the director's vision was absent"
]

# ==================== HELPER FUNCTIONS ====================

def safe_choice(lst: List[str]) -> str:
    """Safely choose a random element from a list."""
    return random.choice(lst) if lst else ""

def format_sentence(template: str, label: int) -> str:
    """Fill in a template with appropriate vocabulary."""
    voc = POS_VOC if label == 1 else NEG_VOC
    
    # Handle plural nouns
    noun = safe_choice(voc["noun"])
    adj = safe_choice(voc["adj"])
    verb = safe_choice(voc["verb"])
    adv = safe_choice(voc["adv"])
    prep = safe_choice(voc.get("prep_phrase", [""]))
    
    # Fill template
    result = template.format(
        adj=adj,
        noun=noun,
        verb=verb,
        adv=adv,
        prep_phrase=prep
    )
    
    return result

def add_variation(text: str, label: int) -> str:
    """Add random variations to make sentences more diverse."""
    
    # 30% chance to add intensifier at the beginning
    if random.random() < 0.3:
        intensifier = random.choice(INTENSIFIERS)
        text = f"{intensifier.capitalize()}, {text.lower()}"
    
    # 20% chance to add conjunction
    if random.random() < 0.2:
        conj = random.choice(["and", "but", "however", "yet"])
        text = f"{text.rstrip('.')} {conj} it was memorable."
    
    # 15% chance to add exclamation
    if random.random() < 0.15:
        text = text.rstrip('.') + "!"
    
    # 10% chance to add "truly" or "really"
    if random.random() < 0.1:
        text = f"Truly, {text.lower()}"
    
    return text

def clean_text(text: str) -> str:
    """Clean and format text properly."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Capitalize first letter
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
    # Ensure proper punctuation
    if text and text[-1] not in string.punctuation:
        text += "."
    return text

# ==================== GENERATION FUNCTIONS ====================

def generate_synthetic_batch(size: int, label: int) -> List[Dict[str, Any]]:
    """Generate synthetic reviews using templates."""
    data = []
    templates = SENTENCE_TEMPLATES_POS if label == 1 else SENTENCE_TEMPLATES_NEG
    
    for _ in range(size):
        template = random.choice(templates)
        sentence = format_sentence(template, label)
        sentence = add_variation(sentence, label)
        sentence = clean_text(sentence)
        
        data.append({"text": sentence, "label": label})
    
    return data

def generate_paraphrased_batch(base_sentences: List[str], size: int, label: int) -> List[Dict[str, Any]]:
    """Generate paraphrased versions of base sentences."""
    data = []
    source = BASE_SENTENCES_POS if label == 1 else BASE_SENTENCES_NEG
    
    for _ in range(size):
        # Choose a base sentence
        s = random.choice(source)
        words = s.lower().split()
        
        # Apply synonym replacements
        new_words = []
        for w in words:
            # Remove punctuation for matching
            clean_w = w.strip(string.punctuation)
            if clean_w in SYNONYMS:
                rep = random.choice(SYNONYMS[clean_w])
                # Preserve original punctuation
                if w.endswith('.') or w.endswith('!') or w.endswith('?'):
                    rep += w[-1]
                new_words.append(rep)
            else:
                new_words.append(w)
        
        new_sent = " ".join(new_words)
        
        # Add suffix based on label
        if label == 1:
            suffixes = [
                " Truly exceptional.",
                " A masterpiece.",
                " Highly worth it.",
                " Absolutely brilliant.",
                " Couldn't recommend more.",
                " A triumph in every way."
            ]
        else:
            suffixes = [
                " Absolutely terrible.",
                " A disaster.",
                " Do not bother.",
                " Completely unwatchable.",
                " A complete waste.",
                " Utterly disappointing."
            ]
        
        new_sent += random.choice(suffixes)
        new_sent = clean_text(new_sent)
        
        data.append({"text": new_sent, "label": label})
    
    return data

def generate_contextual_batch(size: int, label: int) -> List[Dict[str, Any]]:
    """Generate sentences with contextual prefixes."""
    data = []
    prefixes = CONTEXT_PREFIXES_POS if label == 1 else CONTEXT_PREFIXES_NEG
    bases = CONTEXT_BASES_POS if label == 1 else CONTEXT_BASES_NEG
    
    for _ in range(size):
        p = random.choice(prefixes)
        i = random.choice(INTENSIFIERS)
        b = random.choice(bases)
        
        # Vary the structure
        if random.random() < 0.5:
            text = f"{p} {i} {b}."
        else:
            text = f"{p}. {i.capitalize()}, {b}."
        
        text = clean_text(text)
        data.append({"text": text, "label": label})
    
    return data

def generate_noise_robust_batch(size: int, label: int) -> List[Dict[str, Any]]:
    """Generate shorter, more robust sentences."""
    data = []
    adj_list = POS_VOC["adj"] if label == 1 else NEG_VOC["adj"]
    noun_list = POS_VOC["noun"] if label == 1 else NEG_VOC["noun"]
    
    templates = [
        "{n} was {a}.",
        "{a} {n}.",
        "This {n} is {a}.",
        "Very {a} {n}.",
        "The {n} feels {a}.",
        "{a}!",
        "So {a}!",
        "Really {a}.",
        "Just {a}.",
        "Truly {a}."
    ]
    
    for _ in range(size):
        a = random.choice(adj_list)
        n = random.choice(noun_list)
        
        template = random.choice(templates)
        text = template.format(a=a, n=n)
        
        # Small chance of case variation (simulating noise)
        if random.random() > 0.95:
            # Only uppercase a small portion
            mid = len(text) // 2
            text = text[:mid].upper() + text[mid:]
        
        text = clean_text(text)
        data.append({"text": text, "label": label})
    
    return data

def generate_complex_batch(size: int, label: int) -> List[Dict[str, Any]]:
    """Generate more complex, compound sentences."""
    data = []
    templates_pos = [
        "While some films struggle, {noun} {verb} brilliantly and {verb2} the audience.",
        "Not only was the {noun} {adj}, but it also {verb} expectations.",
        "The {noun} {verb} {adv}, making it one of the {adj2} films this year.",
        "Despite the odds, the {noun} managed to {verb} and {verb2} everyone.",
        "The film proves that {noun} can {verb} when done with {adj} skill."
    ]
    
    templates_neg = [
        "While some films succeed, {noun} {verb} terribly and {verb2} the audience.",
        "Not only was the {noun} {adj}, but it also {verb} all hope.",
        "The {noun} {verb} {adv}, making it one of the {adj2} films ever made.",
        "Despite any potential, the {noun} failed to {verb} and {verb2} everyone.",
        "The film proves that {noun} can {verb} when done with {adj} lack of skill."
    ]
    
    templates = templates_pos if label == 1 else templates_neg
    voc = POS_VOC if label == 1 else NEG_VOC
    
    for _ in range(size):
        template = random.choice(templates)
        
        text = template.format(
            noun=random.choice(voc["noun"]),
            verb=random.choice(voc["verb"]),
            verb2=random.choice(voc["verb"]),
            adj=random.choice(voc["adj"]),
            adj2=random.choice(voc["adj"]),
            adv=random.choice(voc["adv"])
        )
        
        text = clean_text(text)
        data.append({"text": text, "label": label})
    
    return data

# ==================== MAIN GENERATION FUNCTION ====================

def run_advanced_generation(output_dir: Path, total_samples: int = 3600):
    """
    Generate synthetic dataset with multiple strategies.
    
    Args:
        output_dir: Directory to save generated files
        total_samples: Total number of samples to generate (default: 3600)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate distribution
    samples_per_strategy = total_samples // 5  # 5 strategies now
    
    datasets_config = {
        "sst2": {"size": samples_per_strategy},
        "imdb": {"size": samples_per_strategy},
        "custom": {"size": samples_per_strategy}
    }
    
    all_combined_data = []
    
    for ds_name, config in datasets_config.items():
        count = config["size"]
        half = count // 2
        
        # Generate with 5 different strategies
        batch1 = generate_synthetic_batch(half, 1) + generate_synthetic_batch(half, 0)
        batch2 = generate_paraphrased_batch([], half, 1) + generate_paraphrased_batch([], half, 0)
        batch3 = generate_contextual_batch(half, 1) + generate_contextual_batch(half, 0)
        batch4 = generate_noise_robust_batch(half, 1) + generate_noise_robust_batch(half, 0)
        batch5 = generate_complex_batch(half, 1) + generate_complex_batch(half, 0)
        
        combined_ds = batch1 + batch2 + batch3 + batch4 + batch5
        df = pd.DataFrame(combined_ds)
        
        # Shuffle the dataframe
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        file_path = output_dir / f"data_{ds_name}.csv"
        df.to_csv(file_path, index=False)
        
        print(f"✅ Generated {len(df)} samples for {ds_name}")
        print(f"   Positive: {df['label'].sum()} | Negative: {len(df) - df['label'].sum()}")
        
        all_combined_data.append(df)
    
    # Combine all datasets
    final_df = pd.concat(all_combined_data, ignore_index=True)
    
    # Shuffle final dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_path = output_dir / "combined_master_dataset.csv"
    final_df.to_csv(final_path, index=False)
    
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY - Synthetic Dataset Generation")
    print("="*60)
    print(f"Total Samples:    {len(final_df)}")
    print(f"Positive (1):    {final_df['label'].sum()}")
    print(f"Negative (0):    {len(final_df) - final_df['label'].sum()}")
    print(f"Master File:     {final_path}")
    print("-"*60)
    print("Sample texts:")
    for i, row in final_df.head(5).iterrows():
        label_str = "✅ Positive" if row['label'] == 1 else "❌ Negative"
        print(f"  [{label_str}] {row['text'][:70]}...")
    print("="*60)

if __name__ == "__main__":
    run_advanced_generation(Path.cwd() / "data", total_samples=3600)