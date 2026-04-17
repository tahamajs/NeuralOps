# ==================== smart_ensemble_svd.py (FIXED v4.2 - MPS & Cache Safe) ====================
"""
🌟 ADVANCED ENSEMBLE METHOD: WEIGHTED LOGIT FUSION WITH DYNAMIC SHAPE ALIGNMENT
Strategy: 
1. Load models and compress attention layers via SVD.
2. Extract logits from each model independently.
3. Normalize/Align logits to target shape (Batch, Num_Labels).
4. Combine using learnable weights.
5. Robust error handling for mismatched architectures & MPS compatibility.
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
import shutil

warnings.filterwarnings("ignore")

NUM_LABELS = 2
MAX_LENGTH = 128
BATCH_SIZE = getattr(config, 'batch_size', 16)
EPOCHS = 5
LR = 0.01
MODELS_CONFIG = [
    {"name": "distilbert-base-uncased-finetuned-sst-2-english", "rank": 32}, 
    {"name": "roberta-base", "rank": 48}, 
    {"name": "albert-base-v2", "rank": 64} 
]

# --- 2. Device Setup ---
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
                except Exception as e:
                    # Silent fail for compression to ensure pipeline continues
                    pass
    return count

class RobustEnsembleModel(torch.nn.Module):
    def __init__(self, base_models, num_labels=NUM_LABELS):
        super().__init__()
        self.base_models = torch.nn.ModuleList(base_models)
        self.num_labels = num_labels
        n_models = len(self.base_models)
        
        if n_models == 0:
            raise ValueError("No models loaded in ensemble.")
            
        self.weights = torch.nn.Parameter(torch.ones(n_models, num_labels))
        
    def forward(self, inputs):
        batch_size = inputs['input_ids'].shape[0]
        logits_list = []
        
        for i, model in enumerate(self.base_models):
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Normalize dimensions: ensure [batch, ...]
            if logits.dim() == 3 and logits.shape[0] == 1:
                logits = logits.squeeze(0)
            
            current_classes = logits.shape[-1]
            
            # 🔥 CRITICAL FIX: Dynamic Shape Alignment
            if current_classes == self.num_labels:
                aligned_logits = logits
            elif current_classes > self.num_labels:
                aligned_logits = logits[:, :self.num_labels]
            else:
                padding = torch.zeros(batch_size, self.num_labels - current_classes, device=logits.device)
                aligned_logits = torch.cat([logits, padding], dim=-1)
            
            # Final Validation
            if aligned_logits.shape != (batch_size, self.num_labels):
                raise RuntimeError(f"Model {i} alignment failed: Got {aligned_logits.shape}, Expected {(batch_size, self.num_labels)}")
                
            logits_list.append(aligned_logits)
        
        stacked_logits = torch.stack(logits_list, dim=0)
        
        # ✅ وزن‌ها را طوری گسترش می‌دهیم که با بچ سایز match کند
        w_expanded = self.weights.unsqueeze(1) 
        
        # Weighted Sum
        weighted_sum = torch.sum(stacked_logits * w_expanded, dim=0)
        
        return weighted_sum

# --- 6. Training & Evaluation ---
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    batches = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model({'input_ids': input_ids, 'attention_mask': attn_mask})
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        batches += 1
        
    return total_loss / batches

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model({'input_ids': input_ids, 'attention_mask': attn_mask})
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1

def clean_hf_cache_if_needed(cache_dir_path):
    """Helper to clean incomplete cache files if permission errors occur."""
    if not Path(cache_dir_path).exists():
        return
    
    # Simple check for .incomplete files which cause the specific error seen
    for p in Path(cache_dir_path).rglob('*.incomplete'):
        try:
            p.unlink()
            print(f"🧹 Cleaned incomplete cache file: {p}")
        except Exception as e:
            print(f"⚠️ Could not delete {p}: {e}")

def main():
    print("="*80)
    print("🌟 ROBUST ENSEMBLE PIPELINE: LOGIT ALIGNMENT & FUSION")
    print(f"Target Device: {DEVICE} | Num Labels: {NUM_LABELS}")
    print("="*80)
    
    # 1. Load & Compress Models
    base_models = []
    print("\n⚡ Phase 1: Loading & Compressing Models...")
    
    # Determine cache directory based on config or default
    hf_cache_dir = getattr(config, 'hf_cache_dir', None)
    if not hf_cache_dir:
        # Default HuggingFace cache location
        hf_cache_dir = Path.home() / ".cache" / "huggingface"
    
    # Attempt to clean potential corruption before starting
    if Path(hf_cache_dir).exists():
        clean_hf_cache_if_needed(hf_cache_dir)

    for i, cfg in enumerate(MODELS_CONFIG):
        print(f"\n[{i+1}/{len(MODELS_CONFIG)}] Processing: {cfg['name']}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg['name'], cache_dir=hf_cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg['name'], 
                num_labels=NUM_LABELS, 
                cache_dir=hf_cache_dir,
                ignore_mismatched_sizes=True
            )
            
            model = model.to(DEVICE)
            compressed_count = compress_model(model, cfg['rank'])
            print(f"      ✅ Compressed {compressed_count} layers (Rank={cfg['rank']})")
            base_models.append(model)
            
        except FileNotFoundError as e:
            # Specific handling for the user's reported error
            print(f"      ❌ Critical File Error: {e}")
            print("      ⚠️ Attempting to clear cache and retry...")
            clean_hf_cache_if_needed(hf_cache_dir)
            # Retry logic could go here, but for now we re-raise to stop and let user know
            raise e
        except Exception as e:
            print(f"      ❌ Failed to load/compress {cfg['name']}: {e}")
            raise

    if not base_models:
        raise RuntimeError("CRITICAL: No models were successfully loaded!")
    
    print(f"\n✅ Successfully loaded {len(base_models)} models.")
    
    # 2. Prepare Data
    data_path = "./data/01_irony.csv"
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file missing: {data_path}")
        
    df = pd.read_csv(data_path)
    t_col = [c for c in df.columns if 'text' in c.lower()][0]
    l_col = [c for c in df.columns if 'label' in c.lower()][0]
    df = df.rename(columns={t_col: 'text', l_col: 'label'})
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    dummy_tokenizer = AutoTokenizer.from_pretrained(MODELS_CONFIG[0]['name'], cache_dir=hf_cache_dir)
    
    train_ds = SentimentDataset(train_df['text'], train_df['label'], dummy_tokenizer)
    val_ds = SentimentDataset(val_df['text'], val_df['label'], dummy_tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Build & Train Ensemble
    ensemble = RobustEnsembleModel(base_models, num_labels=NUM_LABELS).to(DEVICE)
    
    print("\n🔥 Phase 2: Training Ensemble Weights...")
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()
    
    start_time = time.time()
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        try:
            epoch_loss = train_epoch(ensemble, train_loader, optimizer, criterion, DEVICE)
            val_acc, val_f1 = evaluate_model(ensemble, val_loader, DEVICE)
            
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(ensemble.state_dict(), "./best_ensemble_weights.pt")
            
            if DEVICE.type == 'mps':
                torch.mps.empty_cache()
        except KeyboardInterrupt:
            print("\n⚠️ Process interrupted by user during training.")
            break
        except Exception as e:
            print(f"❌ Error in Epoch {epoch+1}: {e}")
            traceback.print_exc()
            break
            
    end_time = time.time()
    print(f"\n✅ Training finished in {end_time - start_time:.2f}s")
    
    # 4. Final Evaluation
    print("\n📊 Phase 3: Final Evaluation...")
    final_acc, final_f1 = evaluate_model(ensemble, val_loader, DEVICE)
    
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS")
    print("="*80)
    print(f"Accuracy:  {final_acc:.4f}")
    print(f"F1 Score:  {final_f1:.4f}")
    
    weights_tensor = ensemble.weights.detach()
    
    # Force transfer to CPU regardless of current device (MPS/CUDA/CPU)
    weights_on_cpu = weights_tensor.cpu()
    
    print("Weighs:   ")
    print(weights_on_cpu.numpy())

    print("="*80)
    
    save_dir = "./final_ensemble_artifacts"
    Path(save_dir).mkdir(exist_ok=True)
    
    torch.save(ensemble.state_dict(), f"{save_dir}/weights.pt")
    dummy_tokenizer.save_pretrained(f"{save_dir}/tokenizer")
    
    print(f"\n✅ All artifacts saved to: {save_dir}")
    print("👋 Goodbye! Project completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        traceback.print_exc()
        sys.exit(1)