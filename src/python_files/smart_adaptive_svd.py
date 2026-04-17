# ==================== smart_adaptive_svd.py (نسخه نهایی - Transformers v4.39+) ====================
"""
🧠 SMART ADAPTIVE SVD OPTIMIZER (FINAL COMPATIBLE VERSION)
Fully compatible with the latest Transformers library (v4.39+).
Fixes: 'eval_strategy' and removed 'tokenizer' arg from Trainer.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from config import config
from svd_compressor import SVDLinearLayer
import time
import gc

# --- 1. Dataset Class ---
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 2. Adaptive Compression Logic ---
def get_adaptive_ranks(model_name, arch_patterns):
    ranks_map = {}
    # Strategy: Higher rank for output layers (64), Medium for Q/K/V (32)
    base_rank_low = 32
    base_rank_high = 64
    
    for name in arch_patterns:
        if 'out' in name.lower() or 'output' in name.lower():
            ranks_map[name] = base_rank_high
        elif 'q_lin' in name.lower() or 'k_lin' in name.lower() or 'v_lin' in name.lower():
            ranks_map[name] = base_rank_low
        else:
            ranks_map[name] = base_rank_high
            
    return ranks_map

def apply_adaptive_svd(model, rank_map):
    print("\n⚙️ Applying Adaptive SVD...")
    compressed_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            matched_pattern = None
            for pattern in rank_map.keys():
                if pattern in name.lower():
                    matched_pattern = pattern
                    break
            
            if matched_pattern:
                rank = rank_map[matched_pattern]
                try:
                    new_layer = SVDLinearLayer(module, rank)
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, new_layer)
                    compressed_count += 1
                    print(f"   ✅ Compressed {name} with Rank {rank}")
                except Exception as e:
                    print(f"   ❌ Failed {name}: {e}")
                    
    return compressed_count

# --- 3. Training Loop (FIXED FOR LATEST TRANSFORMERS) ---
def train_model(model, tokenizer, train_dataset, val_dataset, epochs=3, lr=2e-5):
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='weighted')
        }
        
    # ✅ CRITICAL FIX 1: Use 'eval_strategy' ONLY (No 'evaluation_strategy')
    training_args = TrainingArguments(
        output_dir="./temp_finetune",
        num_train_epochs=epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        
        eval_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        fp16=False,
        remove_unused_columns=False
    )
    
    # ✅ CRITICAL FIX 2: Assign tokenizer to model before creating Trainer
    # In newer versions, Trainer looks for model.tokenizer automatically.
    model.tokenizer = tokenizer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # 'tokenizer' argument is REMOVED in latest versions
        compute_metrics=compute_metrics
    )
    
    print("🔥 Starting Fine-Tuning (Latest Compatible)...")
    start_time = time.time()
    trainer.train()
    results = trainer.evaluate()
    end_time = time.time()
    
    print(f"✅ Fine-Tuning Finished in {end_time - start_time:.2f}s")
    print(f"   Final Accuracy: {results['eval_accuracy']:.4f}")
    print(f"   Final F1 Score: {results['eval_f1']:.4f}")
    
    return results

# --- 4. Main Execution ---
def main():
    print("="*80)
    print("🧠 SMART ADAPTIVE SVD WITH FINE-TUNING (FINAL)")
    print("Goal: Maximize Performance via Custom Ranks + Training")
    print("="*80)
    
    # Load Model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"\n🔄 Loading Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(config.hf_cache_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, cache_dir=str(config.hf_cache_dir)
    )
    model.to(config.device)
    
    # Detect Patterns
    patterns = ['attention.q_lin', 'attention.k_lin', 'attention.v_lin', 'attention.out_lin']
    
    # 1. Get Adaptive Ranks
    rank_map = get_adaptive_ranks(model_name, patterns)
    print(f"\n📋 Assigned Ranks:")
    for pat, r in rank_map.items():
        print(f"   {pat} -> Rank {r}")
    
    # 2. Apply Compression
    count = apply_adaptive_svd(model, rank_map)
    print(f"\n📊 Total Layers Compressed: {count}")
    
    # 3. Prepare Data
    df = pd.read_csv("./data/combined_master_dataset.csv")
    text_col = [c for c in df.columns if 'text' in c.lower()][0]
    label_col = [c for c in df.columns if 'label' in c.lower()][0]
    df = df.rename(columns={text_col: 'text', label_col: 'label'})
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    train_ds = SentimentDataset(train_df['text'], train_df['label'], tokenizer)
    val_ds = SentimentDataset(val_df['text'], val_df['label'], tokenizer)
    
    # 4. Fine-Tune
    final_results = train_model(model, tokenizer, train_ds, val_ds)
    
    # 5. Evaluate Final Performance
    print("\n" + "="*80)
    print("🏆 FINAL PERFORMANCE REPORT")
    print("="*80)
    print(f"Accuracy: {final_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {final_results['eval_f1']:.4f}")
    status = "SUCCESS" if final_results['eval_accuracy'] > 0.95 else "NEEDS IMPROVEMENT"
    print(f"Status: {status}")
    print("="*80)
    
    # Save Best Model
    model.save_pretrained("./smart_adaptive_final_model")
    tokenizer.save_pretrained("./smart_adaptive_final_model")
    print("\n✅ Model saved to './smart_adaptive_final_model'")

if __name__ == "__main__":
    main()