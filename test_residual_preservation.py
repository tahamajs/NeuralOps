import sys
sys.path.insert(0, "Codes")
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from framework.data.cached_tensors import get_cached_tokenized_dataset
from framework.eval_all_models_datasets import apply_module_to_model
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

data = get_cached_tokenized_dataset("sst2", "distilbert", 128)
train_loader = torch.utils.data.DataLoader(data["train"], batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(data["val"], batch_size=128, shuffle=False)

def test_residual_preservation(name, steps=30):
    model = AutoModelForSequenceClassification.from_pretrained("D:/SVD_Project/models/distilbert_sst2")
    if name != "full":
        model, _ = apply_module_to_model(model, name, "attention", rank=16, device=device)
        
        # Apply Residual Identity / Safe Scale Preservation on injected modules
        for module in model.modules():
            if type(module).__name__ == "UniversalAttentionWrapper":
                inner = module.inner
                for p_name, p in inner.named_parameters():
                    # Scale down output projections so residual stream is preserved at init
                    if ("out" in p_name or "proj_out" in p_name or "o_proj" in p_name or "c_proj" in p_name) and "weight" in p_name:
                        with torch.no_grad():
                            p.mul_(0.1)
                            
    model = model.to(device)
    
    # Check step 0 zero-shot
    model.eval()
    preds_0, lbls_0 = [], []
    with torch.no_grad():
        for b_ids, b_mask, b_lbl in val_loader:
            out = model(input_ids=b_ids.to(device), attention_mask=b_mask.to(device))
            preds_0.extend(torch.argmax(out.logits, dim=-1).cpu().tolist())
            lbls_0.extend(b_lbl.tolist())
    acc_0 = accuracy_score(lbls_0, preds_0) * 100.0
    f1_0 = f1_score(lbls_0, preds_0, zero_division=0)
    pos_0 = sum(preds_0) / len(preds_0) * 100.0
    print(f"--> {name:20s} [Step  0]: Acc = {acc_0:6.2f}% | F1 = {f1_0:.4f} | Pos% = {pos_0:5.1f}%", flush=True)
    
    # Train for 30 steps with fine-tuning
    head_params = [p for n, p in model.named_parameters() if ("classifier" in n or "pre_classifier" in n) and p.requires_grad]
    attn_params = [p for n, p in model.named_parameters() if ("inner" in n or "attention" in n) and p.requires_grad]
    base_params = [p for n, p in model.named_parameters() if not ("classifier" in n or "pre_classifier" in n or "inner" in n or "attention" in n) and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": 1e-4, "weight_decay": 0.01},
        {"params": attn_params, "lr": 3e-4, "weight_decay": 0.01},
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
        
    # Evaluate after training
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
    print(f"--> {name:20s} [Step {steps:2d}]: Acc = {acc:6.2f}% | F1 = {f1:.4f} | Pos% = {pos_p:5.1f}%", flush=True)

modules = ["full", "cosformer", "svd_linear", "mamba", "gated_delta_net", "linformer", "hgrn2_outer", "performer"]
for m in modules:
    try:
        test_residual_preservation(m, steps=30)
    except Exception as e:
        print(f"FAILED {m}: {e}", flush=True)
