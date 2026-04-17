# ==================== zero_shot_svd_attention.py (نسخه نهایی - DistilBERT Fix) ====================
import torch
import os
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import time

# Import Config and SVD functions
from config import config
from svd_compressor import compress_bert_true_svd, SVDLinearLayer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def is_attention_layer(name, module):
    """
    Detects Attention layers by name patterns.
    Handles both BERT and DistilBERT architectures.
    """
    name_lower = name.lower()
    
    # Patterns for BERT
    bert_patterns = [
        'attention.self.query', 
        'attention.self.key', 
        'attention.self.value', 
        'attention.output.dense'
    ]
    
    # Patterns for DistilBERT (Based on your debug output)
    distilbert_patterns = [
        'attention.q_lin',
        'attention.k_lin',
        'attention.v_lin',
        'attention.out_lin'
    ]
    
    # Check if any pattern matches
    for pattern in bert_patterns + distilbert_patterns:
        if pattern in name_lower:
            return True
            
    return False

def compress_attention_layers(model, rank):
    compressed_count = 0
    skipped_count = 0
    
    print(f"\n⚡ Applying SVD Compression ONLY on Attention Layers (Rank={rank})...")
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if is_attention_layer(name, module):
                try:
                    original_layer = module
                    new_layer = SVDLinearLayer(original_layer, rank)
                    
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                    
                    setattr(parent, child_name, new_layer)
                    compressed_count += 1
                    print(f"   ✅ Compressed: {name} | Shape: {original_layer.weight.shape}")
                except Exception as e:
                    print(f"   ❌ Failed to compress {name}: {e}")
            else:
                skipped_count += 1
    
    return compressed_count, skipped_count

def main():
    print("="*60)
    print("ZERO-SHOT SVD ON ATTENTION LAYERS (DISTILBERT OPTIMIZED)")
    print("Goal: Replace O(d^2) Attention with O(d*r) without Training")
    print("="*60)
    
    model_name = "distilbert-base-uncased-finetuned-sst-2-english" 
    
    print("\n🔄 Loading Pre-Trained Model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(config.hf_cache_dir))
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            cache_dir=str(config.hf_cache_dir)
        )
        print(f"✅ Loaded: {model_name}")
        print(f"   Model Type: DistilBERT (6 Layers)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model.to(config.device)
    original_params = count_parameters(model)
    
    rank = 32
    compressed_count, skipped_count = compress_attention_layers(model, rank)
    
    if compressed_count == 0:
        print("\n⚠️ WARNING: No layers compressed!")
        return

    compressed_params = count_parameters(model)
    reduction = ((original_params - compressed_params) / original_params) * 100
    
    print(f"\n📊 COMPRESSION SUMMARY:")
    print(f"   Layers Compressed: {compressed_count} (Expected: 24 for DistilBERT)")
    print(f"   Layers Skipped: {skipped_count}")
    print(f"   Params Reduction: {reduction:.2f}%")
    
    # Benchmark Speed
    print("\n⏱️ Measuring Inference Speed...")
    test_text = ["This is a great movie.", "I hate this film."] * 50
    
    start_time = time.time()
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True).to(config.device)
    with torch.no_grad():
        _ = model(**inputs)
    end_time = time.time()
    t_comp = end_time - start_time
    
    print(f"   Compressed Time: {t_comp:.4f}s")
    
    # Theoretical check
    d = 768
    r = 32
    theoretical_ratio = d / r
    print(f"   Theoretical Speedup per Attention MatMul: ~{theoretical_ratio}x")
    
    print("\n" + "="*60)
    print("✅ SUCCESS! Attention layers are now compressed.")
    print(f"   Complexity reduced from O(d^2) to O(d*r) in Attention.")
    print(f"   Total Attention operations reduced significantly.")
    print("="*60)

if __name__ == "__main__":
    main()