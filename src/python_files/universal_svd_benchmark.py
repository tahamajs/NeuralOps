# ==================== universal_svd_benchmark.py ====================
"""
Universal SVD Compression & Benchmarking Tool
Supports: BERT, DistilBERT, RoBERTa, etc.
Features: Auto-detect layers, Multi-Rank Testing, Multi-Dataset Evaluation.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from config import config
from svd_compressor import SVDLinearLayer

# --- 1. تنظیمات و الگوهای لایه‌ها برای مدل‌های مختلف ---
LAYER_PATTERNS = {
    'bert': ['attention.self.query', 'attention.self.key', 'attention.self.value', 'attention.output.dense'],
    'distilbert': ['attention.q_lin', 'attention.k_lin', 'attention.v_lin', 'attention.out_lin'],
    'roberta': ['attention.self.query', 'attention.self.key', 'attention.self.value', 'attention.output.dense'], # Similar to BERT
    'xlnet': ['layer.layer.0.attention.hybrid_attn.q_linear', 'layer.layer.0.attention.hybrid_attn.k_linear', 'layer.layer.0.attention.hybrid_attn.v_linear'] # Simplified check
}

def detect_model_architecture(model_name):
    """Detects model architecture based on name."""
    name_lower = model_name.lower()
    if 'distilbert' in name_lower:
        return 'distilbert'
    elif 'roberta' in name_lower:
        return 'roberta'
    elif 'xlnet' in name_lower:
        return 'xlnet'
    else:
        return 'bert' # Default to BERT patterns

def get_target_patterns(arch):
    return LAYER_PATTERNS.get(arch, LAYER_PATTERNS['bert'])

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

def compress_model(model, rank, layer_patterns):
    """Applies SVD compression to specific attention layers."""
    compressed_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Check if this layer matches any pattern
            is_target = False
            for pattern in layer_patterns:
                if pattern in name.lower():
                    is_target = True
                    break
            
            if is_target:
                try:
                    new_layer = SVDLinearLayer(module, rank)
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, new_layer)
                    compressed_count += 1
                except Exception as e:
                    print(f"   ⚠️ Failed to compress {name}: {e}")
    return compressed_count

def evaluate_model(model, tokenizer, dataset_path, batch_size=32):
    """Evaluates the model on a CSV dataset."""
    df = pd.read_csv(dataset_path)
    
    # Handle column names
    text_col = [c for c in df.columns if 'text' in c.lower()][0]
    label_col = [c for c in df.columns if 'label' in c.lower()][0]
    
    df = df.rename(columns={text_col: 'text', label_col: 'label'})
    
    dataset = SentimentDataset(df['text'], df['label'], tokenizer, max_length=config.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds, all_labels = [], []
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(config.device)
            attn_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits
            
            if logits.shape[-1] == 1:
                preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).long().cpu().numpy()
            else:
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec, rec, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'total_samples': len(y_true)
    }

def run_universal_benchmark():
    print("="*70)
    print("🌍 UNIVERSAL SVD BENCHMARK: MULTI-ARCH & MULTI-RANK TESTING")
    print("="*70)
    
    # Configuration
    models_to_test = [
        "distilbert-base-uncased-finetuned-sst-2-english", # Pre-trained
        "google-bert/bert-base-uncased"                  # Base (will need caution)
    ]
    
    datasets_to_test = ["./data/combined_master_dataset.csv"] # Add more paths here if needed
    
    ranks_to_test = [8, 16, 32, 64]
    
    results_table = []
    
    for model_name in models_to_test:
        print(f"\n🔄 Processing Model: {model_name}")
        arch = detect_model_architecture(model_name)
        patterns = get_target_patterns(arch)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(config.hf_cache_dir))
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=2, cache_dir=str(config.hf_cache_dir)
            )
            model.to(config.device)
        except Exception as e:
            print(f"❌ Could not load model {model_name}: {e}")
            continue
        
        for rank in ranks_to_test:
            # Clone model to avoid modifying original for next iterations
            temp_model = type(model)(model.config).to(config.device)
            temp_model.load_state_dict(model.state_dict())
            
            # Compress
            count = compress_model(temp_model, rank, patterns)
            
            if count == 0:
                print(f"   ⚠️ No layers found for Rank {rank}. Skipping.")
                continue
                
            # Evaluate
            metrics = {}
            for ds_path in datasets_to_test:
                if Path(ds_path).exists():
                    m = evaluate_model(temp_model, tokenizer, ds_path)
                    metrics[Path(ds_path).name] = m
                else:
                    metrics["N/A"] = None
            
            # Store Result
            results_table.append({
                "Model": model_name,
                "Architecture": arch,
                "Rank": rank,
                "Layers Compressed": count,
                "Accuracy": metrics[list(metrics.keys())[0]]['accuracy'] if list(metrics.values())[0] else None,
                "F1 Score": metrics[list(metrics.keys())[0]]['f1'] if list(metrics.values())[0] else None,
                "Status": "Success"
            })
            
            print(f"   ✅ Rank {rank}: Acc={metrics[list(metrics.keys())[0]]['accuracy']:.4f}, F1={metrics[list(metrics.keys())[0]]['f1']:.4f}")

    # Print Summary Table
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY TABLE")
    print("="*70)
    df_results = pd.DataFrame(results_table)
    print(df_results.to_string(index=False))
    
    # Find Best Performance
    best_row = df_results[df_results['Accuracy'].notna()].sort_values(by='Accuracy', ascending=False).iloc[0]
    print("\n🏆 BEST CONFIGURATION FOUND:")
    print(f"   Model: {best_row['Model']}")
    print(f"   Rank: {best_row['Rank']}")
    print(f"   Accuracy: {best_row['Accuracy']:.4f}")
    print(f"   Layers Compressed: {best_row['Layers Compressed']}")
    print("="*70)

if __name__ == "__main__":
    run_universal_benchmark()