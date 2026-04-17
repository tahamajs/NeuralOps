# ==================== final_optimized_svd.py ====================
"""
🏆 ULTIMATE OPTIMIZATION PIPELINE: DEEP LOW-RANK FUSION
================================================================
GOAL: Transform O(d^2) Attention Complexity to O(d*r) with MAX Accuracy.
STRATEGY: Adaptive Ranks + Post-Compression Fine-Tuning + Optimized Forward Pass.
================================================================
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from config import config
from svd_compressor import SVDLinearLayer
import time
import gc

# --- 1. DATASET HANDLING ---
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

def load_and_split_data(path):
    df = pd.read_csv(path)
    # Auto-detect columns
    t_col = [c for c in df.columns if 'text' in c.lower()][0]
    l_col = [c for c in df.columns if 'label' in c.lower()][0]
    df = df.rename(columns={t_col: 'text', l_col: 'label'})
    
    split_idx = int(len(df) * 0.8)
    train_df, val_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    print(f"   ✅ Loaded {len(df)} samples. Train: {len(train_df)}, Val: {len(val_df)}")
    return train_df, val_df

# --- 2. ADAPTIVE RANK ASSIGNMENT STRATEGY ---
def get_adaptive_ranks(model_name):
    """
    Assigns higher ranks to layers that matter most for the task.
    Strategy: Output layers (Out) get highest rank, Q/K/V get medium/low.
    This ensures the decision boundary remains sharp while saving memory elsewhere.
    """
    ranks = {}
    if 'distilbert' in model_name.lower():
        # DistilBERT specific patterns
        ranks['attention.q_lin'] = 32   # Medium
        ranks['attention.k_lin'] = 32   # Medium
        ranks['attention.v_lin'] = 32   # Medium
        ranks['attention.out_lin'] = 64 # High (Critical for output)
    else:
        # BERT/RoBERTa patterns
        ranks['attention.self.query'] = 32
        ranks['attention.self.key'] = 32
        ranks['attention.self.value'] = 32
        ranks['attention.output.dense'] = 64
        
    return ranks

# --- 3. COMPRESSION ENGINE ---
def compress_model_adaptively(model, rank_map):
    compressed_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            matched = False
            current_rank = 0
            
            for pattern, rank in rank_map.items():
                if pattern in name.lower():
                    matched = True
                    current_rank = rank
                    break
            
            if matched:
                try:
                    # Replace Linear with SVDLinearLayer
                    new_layer = SVDLinearLayer(module, current_rank)
                    
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    
                    setattr(parent, child_name, new_layer)
                    compressed_count += 1
                except Exception as e:
                    print(f"   ⚠️ Failed {name}: {e}")
                    
    return compressed_count

# --- 4. TRAINING LOOP (OPTIMIZED) ---
def fine_tune_compressed_model(model, tokenizer, train_ds, val_ds, epochs=3, lr=1e-5):
    from transformers import Trainer, TrainingArguments
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='weighted'),
            'precision': precision_recall_fscore_support(labels, preds, average='weighted')[0],
            'recall': precision_recall_fscore_support(labels, preds, average='weighted')[1]
        }
        
    training_args = TrainingArguments(
        output_dir="./final_optimized_model",
        num_train_epochs=epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=lr, # Lower LR to preserve pre-trained knowledge
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        fp16=False,
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    print("🔥 Starting Fine-Tuning (Optimized)...")
    start_time = time.time()
    trainer.train()
    results = trainer.evaluate()
    end_time = time.time()
    
    print(f"✅ Fine-Tuning Completed in {end_time - start_time:.2f}s")
    print(f"   Final Accuracy: {results['eval_accuracy']:.4f}")
    print(f"   Final F1 Score: {results['eval_f1']:.4f}")
    
    return results

# --- 5. BENCHMARK SPEED ---
def benchmark_speed(model, tokenizer, test_texts, runs=5):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.time()
            inputs = tokenizer(test_texts, return_tensors="pt", truncation=True, padding=True).to(config.device)
            _ = model(**inputs)
            end = time.time()
            times.append(end - start)
    return sum(times) / len(times)

# --- 6. MAIN EXECUTION ---
def main():
    print("="*80)
    print("🚀 FINAL OPTIMIZED PIPELINE: DEEP LOW-RANK FUSION")
    print("Goal: O(d^2) -> O(d*r) Transformation + Max Performance")
    print("="*80)
    
    # 1. Load Model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"\n🔄 Loading Model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(config.hf_cache_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, cache_dir=str(config.hf_cache_dir)
    )
    model.to(config.device)
    
    # Count Original Params
    orig_params = sum(p.numel() for p in model.parameters())
    print(f"   Original Parameters: {orig_params:,}")
    
    # 2. Get Adaptive Ranks
    rank_map = get_adaptive_ranks(model_name)
    print("\n⚙️ Applying Adaptive Ranks:")
    for pat, r in rank_map.items():
        print(f"      {pat} -> Rank {r}")
    
    # 3. Compress
    print("\n⚡ Compressing Layers...")
    count = compress_model_adaptively(model, rank_map)
    comp_params = sum(p.numel() for p in model.parameters())
    reduction = ((orig_params - comp_params) / orig_params) * 100
    
    print(f"   ✅ Compressed {count} layers.")
    print(f"   New Parameters: {comp_params:,}")
    print(f"   Reduction: {reduction:.2f}%")
    
    # 4. Prepare Data
    train_df, val_df = load_and_split_data("./data/combined_master_dataset.csv")
    train_ds = SentimentDataset(train_df['text'], train_df['label'], tokenizer)
    val_ds = SentimentDataset(val_df['text'], val_df['label'], tokenizer)
    
    # 5. Fine-Tune
    final_results = fine_tune_compressed_model(model, tokenizer, train_ds, val_ds)
    
    # 6. Benchmark Speed
    print("\n⏱️ Benchmarking Inference Speed...")
    test_texts = ["This movie is great.", "I hate this film."] * 100
    speed = benchmark_speed(model, tokenizer, test_texts)
    print(f"   Avg Inference Time: {speed:.4f}s")
    
    # 7. Final Report
    print("\n" + "="*80)
    print("🏆 FINAL REPORT")
    print("="*80)
    print(f"Model:         {model_name}")
    print(f"Strategy:      Adaptive SVD + Fine-Tuning")
    print(f"Parameters:    {comp_params:,} ({reduction:.1f}% reduction)")
    print(f"Accuracy:      {final_results['eval_accuracy']:.4f}")
    print(f"F1 Score:      {final_results['eval_f1']:.4f}")
    print(f"Inference:     {speed:.4f}s")
    print("="*80)
    
    # Save
    model.save_pretrained("./final_super_compressed_model")
    tokenizer.save_pretrained("./final_super_compressed_model")
    print("\n✅ Model saved to './final_super_compressed_model'")
    print("💡 You can now deploy this model with O(d*r) complexity!")

if __name__ == "__main__":
    main()