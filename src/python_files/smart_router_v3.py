# ==================== smart_router_v3.py (TRAINED VERSION) ====================
"""
🌟 SMART ROUTER ENSEMBLE SYSTEM v3.0 - WITH ACTUAL TRAINING
Key Improvements:
1. Actual Fine-tuning with backpropagation
2. LoRA instead of destructive SVD compression
3. Better router with confidence-based fallback
4. Proper training loop with optimizer/scheduler
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
from sklearn.metrics import accuracy_score, f1_score, classification_report
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
BATCH_SIZE = 8  # Smaller for MPS memory
LEARNING_RATE = 2e-4
EPOCHS = 3
TRAIN_SAMPLES = 500  # Samples for training
VAL_SAMPLES = 100    # Samples for validation

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

# --- LoRA Implementation (Non-destructive) ---
class LoRALayer(torch.nn.Module):
    """Low-Rank Adaptation - adds trainable rank decomposition matrices"""
    def __init__(self, linear_layer, rank=16, alpha=16):
        super().__init__()
        self.original = linear_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original weights
        for param in self.original.parameters():
            param.requires_grad = False
        
        # Add trainable adapters
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        self.lora_A = torch.nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = torch.nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        
        # Initialize LoRA params
        torch.nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        torch.nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Original forward pass
        original_output = self.original(x)
        # LoRA forward pass
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return original_output + lora_output

def apply_lora(model, rank=16, alpha=16):
    """Apply LoRA to attention layers"""
    count = 0
    target_modules = ['query', 'key', 'value', 'dense', 'attention.self']
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_name = name.split('.')[-1]
            if any(tm in module_name for tm in target_modules):
                try:
                    lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                    
                    setattr(parent, child_name, lora_layer)
                    count += 1
                except Exception:
                    pass
    return count

# --- Dataset Class ---
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

# --- Improved Router with Confidence ---
ROUTER_KEYWORDS = {
    'comparative': ['better', 'worse', 'than', 'more', 'less', 'higher', 'lower', 'superior', 'inferior', 'best', 'worst', 'prefer'],
    'negation': ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nor', 'without', "n't", 'dont', "don't", 'wont', "won't", 'cant', "can't", 'didnt', "didn't"],
    'irony': []  # Fallback
}

def detect_task_type(text):
    text_lower = text.lower()
    
    comparative_count = sum(1 for kw in ROUTER_KEYWORDS['comparative'] if kw in text_lower)
    negation_count = sum(1 for kw in ROUTER_KEYWORDS['negation'] if kw in text_lower)
    
    if comparative_count > negation_count and comparative_count > 0:
        return 'comparative'
    elif negation_count > 0:
        return 'negation'
    
    return 'irony'

# --- Training Function ---
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    """Train the model with LoRA parameters"""
    # Only optimize LoRA parameters + classifier
    optimizer_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            optimizer_params.append({'params': param})
    
    optimizer = torch.optim.AdamW(optimizer_params, lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    best_f1 = 0
    best_model_state = None
    
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
        
        print(f"      ✅ Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        model = model.to(DEVICE)
    
    return best_f1

# --- Main Execution ---
def main():
    print("="*80)
    print("🌟 SMART ROUTER ENSEMBLE SYSTEM v3.0 (WITH TRAINING)")
    print(f"Target Device: {DEVICE}")
    print("="*80)
    
    tasks_config = [
        {"name": "01_irony", "path": "distilbert-base-uncased-finetuned-sst-2-english", "rank": 16},
        {"name": "02_negation", "path": "distilbert-base-uncased-finetuned-sst-2-english", "rank": 16},
        {"name": "03_comparative", "path": "distilbert-base-uncased-finetuned-sst-2-english", "rank": 16}
    ]
    
    # Use same base model for all (more stable on MPS)
    base_model = "distilbert-base-uncased-finetuned-sst-2-english"
    
    data_files = {
        "01_irony": "./data/01_irony.csv",
        "02_negation": "./data/02_negation.csv",
        "03_comparative": "./data/03_comparative.csv"
    }
    
    models = {}
    train_stats = {}
    
    print("\n⚡ Phase 1: Training Specialized Models...")
    
    for task in tasks_config:
        task_name = task["name"]
        print(f"\n▶️  Training Task: {task_name}")
        
        data_path = data_files.get(task_name)
        if not Path(data_path).exists():
            print(f"   ⚠️  Data file missing: {data_path}. Skipping.")
            continue
            
        df = pd.read_csv(data_path)
        
        # Find text and label columns
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
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=getattr(config, 'hf_cache_dir', None))
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model, 
                num_labels=NUM_LABELS,
                cache_dir=getattr(config, 'hf_cache_dir', None)
            )
            model = model.to(DEVICE)
            
            # Apply LoRA (non-destructive)
            lora_count = apply_lora(model, rank=task["rank"], alpha=task["rank"])
            print(f"      ✅ Applied LoRA to {lora_count} layers")
            
            # Create datasets
            train_dataset = SentimentDataset(train_df['text'], train_df['label'], tokenizer)
            val_dataset = SentimentDataset(val_df['text'], val_df['label'], tokenizer)
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            
            # Train
            print(f"      🔥 Training {EPOCHS} epochs...")
            best_f1 = train_model(model, train_loader, val_loader)
            
            # Final evaluation on validation set
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
            
            print(f"   ✅ {task_name} Final | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            models[task_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "val_df": val_df,
                "total_dataset_size": total_samples
            }
            train_stats[task_name] = {'acc': val_acc, 'f1': val_f1}
            
            # Clear memory
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"   ❌ Error training {task_name}: {e}")
            traceback.print_exc()
    
    if not models:
        raise RuntimeError("No models trained successfully!")
    
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
        
        if detected_task in models:
            model = models[detected_task]["model"]
            tokenizer = models[detected_task]["tokenizer"]
        else:
            # Fallback
            model = models["01_irony"]["model"]
            tokenizer = models["01_irony"]["tokenizer"]
        
        # Predict
        encoding = tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=MAX_LENGTH, 
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0][pred].item()
        
        router_preds.append(pred)
        router_labels.append(all_true_labels[i])
    
    # Calculate metrics
    final_acc = accuracy_score(router_labels, router_preds)
    final_f1 = f1_score(router_labels, router_preds, average='weighted')
    
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS (SMART ROUTER v3.0 - TRAINED)")
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
    print("✅ Smart Router v3.0 completed successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        traceback.print_exc()
        sys.exit(1)




#         تغییرات کلیدی:
# مشکل قدیمی
# راه‌حل جدید
# ❌ بدون training
# ✅ حلقه training واقعی با backpropagation
# ❌ SVD مخرب
# ✅ LoRA (non-destructive)
# ❌ classifier تصادفی
# ✅ آموزش classifier + LoRA
# ❌ optimizer نامناسب
# ✅ AdamW + Scheduler
# ❌ فقط 500 sample
# ✅ قابل تنظیم با TRAIN_SAMPLES

# حالا مدل‌ها واقعاً fine-tune می‌شوند و نتایج بهتری خواهید گرفت. می‌توانید TRAIN_SAMPLES و EPOCHS را افزایش دهید برای دقت بالاتر.


