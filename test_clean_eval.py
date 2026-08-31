import sys
from pathlib import Path
sys.path.insert(0, "Codes")
sys.path.insert(0, ".")

import torch
from transformers import AutoModelForSequenceClassification
from framework.data.cached_tensors import get_cached_tokenized_dataset
from framework.eval_all_models_datasets import apply_module_to_model
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

# 1. Load dataset
data = get_cached_tokenized_dataset("sst2", "distilbert", 128)
val_loader = torch.utils.data.DataLoader(data["val"], batch_size=128, shuffle=False)

def eval_mod(name):
    model = AutoModelForSequenceClassification.from_pretrained("D:/SVD_Project/models/distilbert_sst2")
    if name != "full":
        model, _ = apply_module_to_model(model, name, "attention", rank=16, device=device)
    model = model.to(device).eval()
    preds, lbls = [], []
    with torch.no_grad():
        for b_ids, b_mask, b_lbl in val_loader:
            out = model(input_ids=b_ids.to(device), attention_mask=b_mask.to(device))
            preds.extend(torch.argmax(out.logits, dim=-1).cpu().tolist())
            lbls.extend(b_lbl.tolist())
    acc = accuracy_score(lbls, preds) * 100.0
    f1 = f1_score(lbls, preds, zero_division=0)
    pos_p = sum(preds) / len(preds) * 100.0
    print(f"--> {name:20s}: Acc = {acc:6.2f}% | F1 = {f1:.4f} | Pos% = {pos_p:5.1f}%", flush=True)

for m in ["full", "cosformer", "svd_linear", "mamba", "gated_delta_net", "linformer", "hgrn2_outer", "performer"]:
    try:
        eval_mod(m)
    except Exception as e:
        print(f"FAILED {m}: {e}", flush=True)
