"""Normalize model names in master summary to 7 canonical names."""
import csv
import re
from collections import defaultdict

# Canonical models
CANONICAL_MODELS = {
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "gpt2": "gpt2",
    "llama": "llama",
    "qwen": "qwen",
    "t5": "t5",
    "bert": "bert-base-uncased",
}

def normalize_model(name: str) -> str:
    """Map any model variant to canonical name."""
    if not name:
        return ""
    
    # Remove Windows paths
    name = re.sub(r"^[A-Z]:\\", "", name, flags=re.IGNORECASE)
    name = name.replace("\\", "/")
    
    # Remove leading directories
    if "/" in name:
        name = name.split("/")[-1]
    
    name_lower = name.lower()
    
    # Match against canonical
    for key, canonical in CANONICAL_MODELS.items():
        if key in name_lower:
            return canonical
    
    return name  # Unknown

if __name__ == "__main__":
    rows = list(csv.DictReader(open("results/_MASTER_SUMMARY.csv")))
    
    # Save before/after
    before = set(r["model"] for r in rows if r.get("model"))
    
    for r in rows:
        r["model"] = normalize_model(r.get("model", ""))
    
    after = set(r["model"] for r in rows if r.get("model"))
    
    # Write back
    with open("results/_MASTER_SUMMARY.csv", "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Models before: {len(before)}")
    print(f"Models after: {len(after)}")
    print("\nAfter:")
    for m in sorted(after):
        print(f"  {m}")
