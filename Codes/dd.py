#!/usr/bin/env python3
"""
Download all required transformer models to a local cache directory.
Run once before starting experiments to ensure local files are available.
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# List of models to download
MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    
]

# Local cache directory (absolute or relative)
CACHE_DIR = "../models"

def download_model(model_name: str, cache_dir: str):
    """Download tokenizer, config, and model for a given name."""
    print(f"\n=== Downloading {model_name} ===")
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"  - Tokenizer saved")
        # Download config
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        print(f"  - Config saved")
        # Download model (with classification head – you can also use AutoModel if preferred)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, cache_dir=cache_dir)
        print(f"  - Model saved")
        print(f"✅ {model_name} downloaded successfully to {cache_dir}")
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")

def main():
    # Create cache directory if it doesn't exist
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Models will be cached in: {os.path.abspath(CACHE_DIR)}")
    for model in MODELS:
        download_model(model, CACHE_DIR)
    print("\n🎉 All models downloaded. You can now set HF_HOME and use local_files_only=True.")

if __name__ == "__main__":
    main()