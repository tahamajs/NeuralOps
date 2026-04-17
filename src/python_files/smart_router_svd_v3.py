# ==================== smart_router_svd_v3.py ====================
"""
🌟 SMART ROUTER ENSEMBLE SYSTEM v3.0 - SVD + TRAINING
Strategy:
1. Load pretrained models
2. Fine-tune classifier + train on data
3. Apply SVD compression AFTER training
4. Evaluate on full dataset
Author: RAINO
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from config import config
import time
import warnings
import sys
import traceback
import gc

warnings.filterwarnings("ignore")

# --- Configuration ---
NUM_LABELS = 2
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
EPOCHS = 3
TRAIN_SAMPLES = 500

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
            return torch.device("cpu")
    return torch.device("cpu")

DEVICE = get_device()
print(f"🚀 System Ready | Device: {DEVICE}")

# --- SVD Compressor (Your Original) ---
class SVDLinearLayer(torch.nn.Module):
    def __init__(self, linear_layer, rank):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        
        original_weight = linear_layer.weight.data
        U, S, Vh = torch.linalg.svd(original_weight, full_matrices=False)
        
        self.U = torch.nn.Parameter(U[:, :rank])
        self.S = torch.nn.Parameter(S[:rank])
        self.Vh = torch.nn.Parameter(Vh[:rank, :])
        
        if linear_layer.bias is not None:
            self.bias = torch.nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.bias = None
            
    def forward(self, x):
        reconstructed = self.U @ (self.S.unsqueeze(1) * self.Vh)
        return torch.nn.functional.linear(x, reconstructed, self.bias)

def compress_model(model, rank):
    """Apply SVD compression to model layers"""
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

# --- Dataset ---
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
            text, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Router ---
ROUTER_KEYWORDS = {
    'comparative': ['better', 'worse', 'than', 'more', 'less', 'higher', 'lower', 'superior', 'inferior', 'best', 'worst', 'prefer'],
    'negation': ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nor', 'without', "n't", 'dont', "don't", 'wont', "won't", 'cant', "can't"],
    'irony': []
}

def detect_task_type(text):
    text_lower = text.lower()
    comp_count = sum(1 for kw in ROUTER_KEYWORDS['comparative'] if kw in text_lower)
    neg_count = sum(1 for kw in ROUTER_KEYWORDS['negation'] if kw in text_lower)
    
    if comp_count > neg_count and comp_count > 0:
        return 'comparative'
    elif neg_count > 0:
        return 'negation'
    return 'irony'

# --- Training Function ---
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """Train the model (classifier + backbone)"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    best_f1 = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"      Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        preds, true_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                true_labels.extend(labels.numpy())
        
        val_acc = accuracy_score(true_labels, preds)
        val_f1 = f1_score(true_labels, preds, average='weighted')
        
        print(f"      ✅ Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
    
    return best_f1

# --- Main ---
def main():
    print("="*80)
    print("🌟 SMART ROUTER ENSEMBLE SYSTEM v3.0 (SVD + TRAINING)")
    print(f"Target Device: {DEVICE}")
    print("="*80)
    
    # Use same base model for stability
    base_model = "distilbert-base-uncased-finetuned-sst-2-english"
    
    tasks_config = [
        {"name": "01_irony", "rank": 32},
        {"name": "02_negation", "rank": 32},
        {"name": "03_comparative", "rank": 32}
    ]
    
    data_files = {
        "01_irony": "./data/01_irony.csv",
        "02_negation": "./data/02_negation.csv",
        "03_comparative": "./data/03_comparative.csv"
    }
    
    models = {}
    train_stats = {}
    
    print("\n⚡ Phase 1: Training Specialized Models (Fine-tune THEN Compress)...")
    
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
        print(f"   📊 Total samples: {total_samples}")
        
        # Split data
        df_train = df.head(TRAIN_SAMPLES)
        train_df = df_train.iloc[:int(0.8*len(df_train))].reset_index(drop=True)
        val_df = df_train.iloc[int(0.8*len(df_train)):].reset_index(drop=True)
        
        print(f"   📊 Train: {len(train_df)}, Val: {len(val_df)}")
        
        try:
            # Step 1: Load pretrained model
            print(f"   📥 Loading pretrained model...")
            tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=getattr(config, 'hf_cache_dir', None))
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model, 
                num_labels=NUM_LABELS,
                cache_dir=getattr(config, 'hf_cache_dir', None)
            )
            model = model.to(DEVICE)
            
            # Step 2: Train BEFORE compression
            print(f"   🔥 Phase A: Training BEFORE SVD compression...")
            train_dataset = SentimentDataset(train_df['text'], train_df['label'], tokenizer)
            val_dataset = SentimentDataset(val_df['text'], val_df['label'], tokenizer)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            
            best_f1_before = train_model(model, train_loader, val_loader)
            
            # Evaluate before SVD
            model.eval()
            preds, true_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                    true_labels.extend(batch['labels'].numpy())
            
            acc_before = accuracy_score(true_labels, preds)
            f1_before = f1_score(true_labels, preds, average='weighted')
            print(f"   ✅ Pre-SVD | Val Acc: {acc_before:.4f} | F1: {f1_before:.4f}")
            
            # Step 3: Apply SVD AFTER training
            print(f"   🗜️  Phase B: Applying SVD compression (rank={task['rank']})...")
            compressed_count = compress_model(model, task['rank'])
            print(f"      ✅ Compressed {compressed_count} layers")
            
            # Evaluate after SVD
            model.eval()
            preds, true_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                    true_labels.extend(batch['labels'].numpy())
            
            acc_after = accuracy_score(true_labels, preds)
            f1_after = f1_score(true_labels, preds, average='weighted')
            print(f"   ✅ Post-SVD | Val Acc: {acc_after:.4f} | F1: {f1_after:.4f}")
            print(f"   📉 SVD Impact: {acc_after - acc_before:+.4f}")
            
            models[task_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "val_df": val_df,
                "total_dataset_size": total_samples
            }
            train_stats[task_name] = {'acc': acc_after, 'f1': f1_after, 'acc_before': acc_before}
            
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            traceback.print_exc()
    
    if not models:
        raise RuntimeError("No models trained!")
    
    print("\n🔥 Phase 2: Testing Smart Router on FULL DATASETS...")
    
    # Collect all data
    all_texts = []
    all_true_labels = []
    all_source_tasks = []
    
    for task_name in models.keys():
        full_df = pd.read_csv(data_files[task_name])
        t_col = [c for c in full_df.columns if 'text' in c.lower()][0]
        l_col = [c for c in full_df.columns if 'label' in c.lower()][0]
        full_df = full_df.rename(columns={t_col: 'text', l_col: 'label'})
        
        all_texts.extend(full_df['text'].tolist())
        all_true_labels.extend(full_df['label'].tolist())
        all_source_tasks.extend([task_name] * len(full_df))
    
    print(f"   Total Test Samples: {len(all_texts)}")
    
    # Shuffle
    indices = np.random.permutation(len(all_texts))
    all_texts = [all_texts[i] for i in indices]
    all_true_labels = [all_true_labels[i] for i in indices]
    all_source_tasks = [all_source_tasks[i] for i in indices]
    
    # Run Router
    router_preds = []
    router_labels = []
    
    print("   Running inference...")
    for i, text in enumerate(all_texts):
        if i % 500 == 0:
            print(f"      Progress: {i}/{len(all_texts)}")
        
        detected_task = detect_task_type(text)
        
        if detected_task not in models:
            detected_task = '01_irony'  # Fallback
        
        model = models[detected_task]["model"]
        tokenizer = models[detected_task]["tokenizer"]
        
        encoding = tokenizer(
            text, truncation=True, padding='max_length', 
            max_length=MAX_LENGTH, return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=-1).item()
        
        router_preds.append(pred)
        router_labels.append(all_true_labels[i])
    
    # Final metrics
    final_acc = accuracy_score(router_labels, router_preds)
    final_f1 = f1_score(router_labels, router_preds, average='weighted')
    
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS (SMART ROUTER - SVD + TRAINED)")
    print("="*80)
    print(f"Total Samples: {len(router_preds)}")
    print(f"Overall Accuracy:  {final_acc:.4f}")
    print(f"Overall F1 Score:  {final_f1:.4f}")
    
    # Breakdown
    print("\n📊 Breakdown by Detected Task:")
    for task_name in models.keys():
        task_indices = [i for i, src in enumerate(all_source_tasks) if src == task_name]
        if task_indices:
            t_preds = [router_preds[i] for i in task_indices]
            t_labels = [router_labels[i] for i in task_indices]
            t_acc = accuracy_score(t_labels, t_preds)
            t_f1 = f1_score(t_labels, t_preds, average='weighted')
            print(f"   {task_name}: Acc={t_acc:.4f}, F1={t_f1:.4f} (Count: {len(task_indices)})")
    
    # Comparison
    print("\n📈 Comparison: Router vs Best Single Task")
    best_single = max(train_stats.items(), key=lambda x: x[1]['f1'])
    improvement = final_acc - best_single[1]['acc']
    print(f"   Best Single ({best_single[0]}): {best_single[1]['acc']:.4f}")
    print(f"   Router System:              {final_acc:.4f}")
    print(f"   Improvement:                {improvement:+.4f}")
    
    print("="*80)
    print("✅ Smart Router v3.0 (SVD) completed!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)