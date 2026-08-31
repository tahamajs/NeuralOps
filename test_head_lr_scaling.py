import sys
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

def test_module_lr(module_name, steps=100, lr_head=1e-3, lr_attn=3e-4, lr_base=3e-5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n=================== TESTING {module_name} with Param Group LRs (head={lr_head}, attn={lr_attn}, base={lr_base}) ===================", flush=True)
    
    # 1. Load base model
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    # 2. Inject module
    if module_name != "full":
        model, _ = apply_module_to_model(model, module_name, "attention", rank=16, device=device)
        
    model = model.to(device)
    
    # 3. Param groups with discriminative learning rates:
    head_params, attn_params, base_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in name or "pre_classifier" in name:
            head_params.append(p)
        elif "inner" in name or "attention" in name:
            attn_params.append(p)
        else:
            base_params.append(p)
            
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": lr_head, "weight_decay": 0.01},
        {"params": attn_params, "lr": lr_attn, "weight_decay": 0.01},
        {"params": base_params, "lr": lr_base, "weight_decay": 0.01},
    ])
    
    # 4. Load dataset
    cached_splits = get_cached_tokenized_dataset("sst2", "distilbert", 128)
    train_loader = torch.utils.data.DataLoader(cached_splits["train"], batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(cached_splits["val"], batch_size=128, shuffle=False)
    
    # 5. Train
    model.train()
    step_cnt = 0
    for epoch in range(5):
        for b_ids, b_mask, b_lbl in train_loader:
            if step_cnt >= steps:
                break
            optimizer.zero_grad()
            out = model(input_ids=b_ids.to(device), attention_mask=b_mask.to(device), labels=b_lbl.to(device))
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step_cnt % 20 == 0:
                preds_train = torch.argmax(out.logits, dim=-1)
                p_train_pos = (preds_train == 1).float().mean().item() * 100.0
                print(f"  Step {step_cnt:3d} | Loss: {loss.item():.4f} | Train Pos%: {p_train_pos:5.1f}%", flush=True)
                
            step_cnt += 1
        if step_cnt >= steps:
            break
            
    # 6. Evaluate
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for b_ids, b_mask, b_lbl in val_loader:
            out = model(input_ids=b_ids.to(device), attention_mask=b_mask.to(device))
            logits = out.logits
            preds = torch.argmax(logits, dim=-1)
            val_preds.extend(preds.cpu().numpy().tolist())
            val_labels.extend(b_lbl.numpy().tolist())
            
    p_pos = sum(val_preds) / len(val_preds) * 100.0
    acc = accuracy_score(val_labels, val_preds) * 100.0
    f1 = f1_score(val_labels, val_preds, zero_division=0)
    print(f"--> FINAL {module_name:15s} (Step {steps}): Acc={acc:.2f}%, F1={f1:.4f}, Pos%={p_pos:.1f}%", flush=True)
    return acc, f1

if __name__ == "__main__":
    test_modules = ["full", "mamba", "gated_delta_net", "linformer", "hgrn2_outer", "performer"]
    for mod in test_modules:
        try:
            test_module_lr(mod, steps=60, lr_head=1e-3, lr_attn=3e-4, lr_base=3e-5)
        except Exception as e:
            print(f"Error {mod}: {e}", flush=True)
