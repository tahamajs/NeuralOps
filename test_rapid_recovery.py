import sys
sys.path.insert(0, "Codes")
sys.path.insert(0, ".")

import torch
from transformers import AutoModelForSequenceClassification
from framework.data.cached_tensors import get_cached_tokenized_dataset
from framework.eval_all_models_datasets import apply_module_to_model
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

data = get_cached_tokenized_dataset("sst2", "distilbert", 128)
train_loader = torch.utils.data.DataLoader(data["train"], batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(data["val"], batch_size=128, shuffle=False)

def train_and_eval(name, steps=30):
    model = AutoModelForSequenceClassification.from_pretrained("D:/SVD_Project/models/distilbert_sst2")
    if name != "full":
        model, _ = apply_module_to_model(model, name, "attention", rank=16, device=device)
    model = model.to(device)
    
    # Param groups: head & attention learn faster than backbone
    head_params = [p for n, p in model.named_parameters() if ("classifier" in n or "pre_classifier" in n) and p.requires_grad]
    attn_params = [p for n, p in model.named_parameters() if ("inner" in n or "attention" in n) and p.requires_grad]
    base_params = [p for n, p in model.named_parameters() if not ("classifier" in n or "pre_classifier" in n or "inner" in n or "attention" in n) and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": 2e-4, "weight_decay": 0.01},
        {"params": attn_params, "lr": 5e-4, "weight_decay": 0.01},
        {"params": base_params, "lr": 3e-5, "weight_decay": 0.01},
    ])
    
    model.train()
    step_cnt = 0
    for b_ids, b_mask, b_lbl in train_loader:
        if step_cnt >= steps:
            break
        optimizer.zero_grad()
        out = model(input_ids=b_ids.to(device), attention_mask=b_mask.to(device), labels=b_lbl.to(device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_cnt += 1
        
    model.eval()
    preds, lbls = [], []
    with torch.no_grad():
        for b_ids, b_mask, b_lbl in val_loader:
            out = model(input_ids=b_ids.to(device), attention_mask=b_mask.to(device))
            preds.extend(torch.argmax(out.logits, dim=-1).cpu().tolist())
            lbls.extend(b_lbl.tolist())
            
    acc = accuracy_score(lbls, preds) * 100.0
    f1 = f1_score(lbls, preds, zero_division=0)
    pos_p = sum(preds) / len(preds) * 100.0
    print(f"--> {name:20s} (After {steps} steps): Acc = {acc:6.2f}% | F1 = {f1:.4f} | Pos% = {pos_p:5.1f}%", flush=True)

for m in ["mamba", "gated_delta_net", "linformer", "hgrn2_outer"]:
    try:
        train_and_eval(m, steps=30)
    except Exception as e:
        print(f"FAILED {m}: {e}", flush=True)
