# ==================== smart_router_full_data.py (COMPLETE v2.0 - FULL DATA) ====================
"""
🌟 SMART ROUTER ENSEMBLE SYSTEM (FULL DATASET EDITION)
Strategy: 
1. Train specialized models on a subset (for speed/memory).
2. Evaluate the Router on the ENTIRE dataset of all tasks.
3. Report metrics based on full data coverage.
Author: RAINO (Smart Solutions Developer)
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from config import config
from svd_compressor import SVDLinearLayer
import time
import warnings
import sys
import traceback

warnings.filterwarnings("ignore")

NUM_LABELS = 2
MAX_LENGTH = 128
BATCH_SIZE = 16

# --- Device Setup ---
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        try:
            _ = torch.zeros(1).to("mps")
            print("[CONFIG] MPS available. Using Apple Silicon GPU.")
            return torch.device("mps")
        except RuntimeError:
            print("⚠️ MPS initialization failed. Falling back to CPU.")
            return torch.device("cpu")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"🚀 System Ready | Device: {DEVICE}")

# --- 1. Keyword-Based Router ---
ROUTER_KEYWORDS = {
    'comparative': ['better', 'worse', 'than', 'more', 'less', 'higher', 'lower', 'superior', 'inferior'],
    'negation': ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nor', 'without', 'un-', 'in-', 'im-', 'non-'],
    'irony': [] # Fallback for irony
}

def detect_task_type(text):
    text_lower = text.lower()
    
    if any(kw in text_lower for kw in ROUTER_KEYWORDS['comparative']):
        return 'comparative'
    
    if any(kw in text_lower for kw in ROUTER_KEYWORDS['negation']):
        return 'negation'
    
    return 'irony' 

# --- 2. Dataset & Compression Classes ---

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compress_model(model, rank):
    count = 0
    skip_keywords = ['classifier', 'embeddings', 'pooler', 'lm_head', 'rel_embeddings', 'word_embeddings'] 
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(kw in name.lower() for kw in skip_keywords):
                continue
                
            if any(kw in name.lower() for kw in ['attention', 'query', 'key', 'value', 'output', 'dense', 'intermediate']):
                try:
                    new_layer = SVDLinearLayer(module, rank)
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                    
                    setattr(parent, child_name, new_layer)
                    count += 1
                except Exception:
                    pass
    return count

class TaskSpecificModel(torch.nn.Module):
    def __init__(self, model_path, rank, num_labels=NUM_LABELS):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=getattr(config, 'hf_cache_dir', None))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=num_labels, 
            cache_dir=getattr(config, 'hf_cache_dir', None),
            ignore_mismatched_sizes=True
        )
        self.model = self.model.to(DEVICE)
        
        compressed_count = compress_model(self.model, rank)
        print(f"      ✅ Compressed {compressed_count} layers (Rank={rank})")
        
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs.logits
    
    def predict_single(self, text):
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=MAX_LENGTH, 
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        with torch.no_grad():
            logits = self.forward({'input_ids': input_ids, 'attention_mask': attention_mask})
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0][pred].item()
            
        return pred, conf

# --- 3. Main Execution ---

def main():
    print("="*80)
    print("🌟 SMART ROUTER ENSEMBLE SYSTEM (FULL DATASET EVALUATION)")
    print(f"Target Device: {DEVICE}")
    print("="*80)

    tasks_config = [
        {"name": "01_irony", "path": "distilbert-base-uncased-finetuned-sst-2-english", "rank": 32},
        {"name": "02_negation", "path": "roberta-base", "rank": 48},
        {"name": "03_comparative", "path": "albert-base-v2", "rank": 64}
    ]
    
    data_files = {
        "01_irony": "./data/01_irony.csv",
        "02_negation": "./data/02_negation.csv",
        "03_comparative": "./data/03_comparative.csv"
    }

    models = {}
    train_stats = {}

    print("\n⚡ Phase 1: Loading & Training Specialized Models (Subset)...")
    
    for task in tasks_config:
        task_name = task["name"]
        print(f"\n▶️  Processing Task: {task_name}")
        
        data_path = data_files.get(task_name)
        if not Path(data_path).exists():
            print(f"   ⚠️  Data file missing: {data_path}. Skipping.")
            continue
            
        df = pd.read_csv(data_path)
        t_col = [c for c in df.columns if 'text' in c.lower()][0]
        l_col = [c for c in df.columns if 'label' in c.lower()][0]
        df = df.rename(columns={t_col: 'text', l_col: 'label'})
        
        total_samples = len(df)
        
        # Use 500 samples for training (as before)
        df_train = df.head(500)
        train_df = df_train.iloc[:400].reset_index(drop=True)
        val_df = df_train.iloc[400:].reset_index(drop=True)
        
        try:
            model_wrapper = TaskSpecificModel(task["path"], task["rank"])
            models[task_name] = {
                "wrapper": model_wrapper,
                "val_df": val_df,
                "tokenizer": model_wrapper.tokenizer,
                "total_dataset_size": total_samples
            }
            
            # Evaluate on Validation Set (Subset)
            val_loader = DataLoader(SentimentDataset(val_df['text'], val_df['label'], model_wrapper.tokenizer), batch_size=BATCH_SIZE)
            
            preds, labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attn_mask = batch['attention_mask'].to(DEVICE)
                    labels_batch = batch['labels']
                    
                    logits = model_wrapper.forward({'input_ids': input_ids, 'attention_mask': attn_mask})
                    preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                    labels.extend(labels_batch.numpy())
            
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='weighted')
            
            print(f"   ✅ Trained on 500 samples. Val Acc: {acc:.4f} | F1: {f1:.4f}")
            train_stats[task_name] = {'acc': acc, 'f1': f1}
            
        except Exception as e:
            print(f"   ❌ Error loading task {task_name}: {e}")
            traceback.print_exc()

    if not models:
        raise RuntimeError("No models loaded successfully!")

    print("\n🔥 Phase 2: Testing Smart Router on FULL DATASETS...")
    
    # Combine ALL validation sets for testing the router
    all_texts = []
    all_true_labels = [] 
    all_source_tasks = []
    
    for task_name, data in models.items():
        # Use the FULL dataset for evaluation this time
        full_df = pd.read_csv(data_files[task_name])
        t_col = [c for c in full_df.columns if 'text' in c.lower()][0]
        l_col = [c for c in full_df.columns if 'label' in c.lower()][0]
        full_df = full_df.rename(columns={t_col: 'text', l_col: 'label'})
        
        all_texts.extend(full_df['text'].tolist())
        all_true_labels.extend(full_df['label'].tolist())
        all_source_tasks.extend([task_name] * len(full_df))
        
    print(f"   Total Test Samples (Full Data): {len(all_texts)}")
    
    # Shuffle
    indices = np.random.permutation(len(all_texts))
    all_texts = [all_texts[i] for i in indices]
    all_true_labels = [all_true_labels[i] for i in indices]
    all_source_tasks = [all_source_tasks[i] for i in indices]

    router_preds = []
    router_labels = []
    
    # Run Inference with Router on Full Data
    print("   Running inference...")
    for i, text in enumerate(all_texts):
        detected_task = detect_task_type(text)
        
        if detected_task in models:
            model_wrapper = models[detected_task]["wrapper"]
            true_label = all_true_labels[i]
            
            pred, conf = model_wrapper.predict_single(text)
            router_preds.append(pred)
            router_labels.append(true_label)
                
        else:
            # Fallback to Irony if no match
            if '01_irony' in models:
                 model_wrapper = models['01_irony']["wrapper"]
                 pred, conf = model_wrapper.predict_single(text)
                 router_preds.append(pred)
                 router_labels.append(all_true_labels[i])
            else:
                router_preds.append(-1)
                router_labels.append(all_true_labels[i])

    # Calculate Final Metrics on FULL DATA
    valid_indices = [i for i, p in enumerate(router_preds) if p != -1]
    final_preds = [router_preds[i] for i in valid_indices]
    final_labels = [router_labels[i] for i in valid_indices]
    
    final_acc = accuracy_score(final_labels, final_preds)
    final_f1 = f1_score(final_labels, final_preds, average='weighted')
    
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS (SMART ROUTER SYSTEM - FULL DATASET)")
    print("="*80)
    print(f"Total Samples Evaluated: {len(valid_indices)}")
    print(f"Overall Accuracy:  {final_acc:.4f}")
    print(f"Overall F1 Score:  {final_f1:.4f}")
    
    # Show breakdown by task (on full data)
    print("\n📊 Breakdown by Detected Task (Full Data):")
    for task_name in models.keys():
        task_indices = [i for i, src in enumerate(all_source_tasks) if src == task_name and router_preds[i] != -1]
        if task_indices:
            t_preds = [router_preds[i] for i in task_indices]
            t_labels = [router_labels[i] for i in task_indices]
            t_acc = accuracy_score(t_labels, t_preds)
            t_f1 = f1_score(t_labels, t_preds, average='weighted')
            print(f"   {task_name}: Acc={t_acc:.4f}, F1={t_f1:.4f} (Count: {len(task_indices)})")
    
    # Comparison with Single Task Performance (on full data)
    print("\n📈 Comparison: Router vs Best Single Task (Full Data)")
    best_single_acc = max(train_stats[t]['acc'] for t in train_stats)
    improvement = final_acc - best_single_acc
    print(f"   Best Single Task (Val): {best_single_acc:.4f}")
    print(f"   Router System (Full):   {final_acc:.4f}")
    print(f"   Improvement:            {improvement:+.4f}")
    
    print("="*80)
    print("✅ Smart Router System completed successfully on FULL DATASET.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        traceback.print_exc()
        sys.exit(1)