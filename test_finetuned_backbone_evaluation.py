import sys
import os
from pathlib import Path
proj_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(proj_dir / "Codes"))
sys.path.insert(0, str(proj_dir))

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from framework.eval_all_models_datasets import apply_module_to_model
from framework.data.cached_tensors import get_cached_tokenized_dataset
from sklearn.metrics import accuracy_score, f1_score

def test_module(module_name, model_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n---> Testing {module_name} on {model_id}...")
    
    # 1. Load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    
    # 2. Inject module
    if module_name != "full":
        model, _ = apply_module_to_model(model, module_name, "attention", rank=16, device=device)
        
    model = model.to(device)
    model.eval()
    
    # 3. Load dataset
    cached_splits = get_cached_tokenized_dataset("sst2", "distilbert", 128)
    val_loader = torch.utils.data.DataLoader(cached_splits["val"], batch_size=128, shuffle=False)
    
    # 4. Zero-shot evaluation
    val_preds, val_labels = [], []
    with torch.no_grad():
        for b_ids, b_mask, b_lbl in val_loader:
            out = model(input_ids=b_ids.to(device), attention_mask=b_mask.to(device))
            logits = out.logits
            preds = torch.argmax(logits, dim=-1)
            val_preds.extend(preds.cpu().numpy().tolist())
            val_labels.extend(b_lbl.numpy().tolist())
            
    acc = accuracy_score(val_labels, val_preds) * 100.0
    f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
    pos_pct = sum(val_preds) / len(val_preds) * 100.0
    print(f"--> {module_name:20s}: Accuracy = {acc:.2f}%, F1 = {f1:.4f}, Pos% = {pos_pct:.1f}%")
    return acc, f1

if __name__ == "__main__":
    modules = ["full", "mamba", "gated_delta_net", "linformer", "hgrn2_outer", "performer", "cosformer", "svd_linear"]
    for mod in modules:
        try:
            test_module(mod)
        except Exception as e:
            print(f"Error testing {mod}: {e}")
