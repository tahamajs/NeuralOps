import os
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import config

def load_original_model():
    cache_dir = Path(config.hf_cache_dir)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created Hugging Face cache directory at: {cache_dir}")

    model_name = config.model_name
    num_labels = config.num_labels
    
    print(f"🚀 Loading model '{model_name}' with {num_labels} labels...")
    print(f"   Cache Dir: {cache_dir}")
    print(f"   Device: {config.device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=False
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=str(cache_dir),
            local_files_only=False
        ).to(config.device)
        
        print("✅ Model and Tokenizer loaded successfully.")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Error loading model from {model_name}. Check your internet connection or mirror status.")

def save_model(model, path):
    output_path = Path(path)
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directory: {output_path}")
    
    print(f"💾 Saving model to: {output_path}")
    
    try:
        model.save_pretrained(output_path)
        print("✅ Model saved successfully.")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        raise

if __name__ == "__main__":
    try:
        model, tokenizer = load_original_model()
        
        test_text = "This is a test sentence for verification."
        inputs = tokenizer(test_text, return_tensors="pt").to(config.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"\n🧪 Quick Test Prediction Logits: {outputs.logits[0].tolist()}")
            
        print("\n✅ Model loading test completed successfully.")
        
    except Exception as e:
        print(f"\n❌ Model loading test failed: {e}")
        raise