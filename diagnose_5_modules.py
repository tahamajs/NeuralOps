import sys
from pathlib import Path
proj_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(proj_dir / "Codes"))
sys.path.insert(0, str(proj_dir))

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.eval_all_models_datasets import apply_module_to_model
from framework.data.cached_tensors import get_cached_tokenized_dataset
from sklearn.metrics import accuracy_score, f1_score

def diagnose_module(module_name, model_id="distilbert-base-uncased", steps=20, lr=3e-5):
    print(f"\n=================== DIAGNOSING {module_name} on {model_id} ===================")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    
    # Check head init
    print("Initial classification head weights:")
    for name, p in model.named_parameters():
        if "classifier" in name:
            print(f"  {name:30s}: mean={p.data.mean().item():.5f}, std={p.data.std().item():.5f}, requires_grad={p.requires_grad}")
            
    # 2. Inject module
    if module_name != "full":
        model, _ = apply_module_to_model(model, module_name, "attention", rank=16, device=device)
        
    model = model.to(device)
    
    # 3. Check parameter requires_grad and optimizer inclusion
    params_in_model = list(model.parameters())
    trainable_params = [p for p in params_in_model if p.requires_grad]
    print(f"Total params: {len(params_in_model)}, Trainable params: {len(trainable_params)}")
    
    injected_named_params = []
    for name, p in model.named_parameters():
        if "inner" in name or "attention" in name:
            injected_named_params.append((name, p))
    print(f"Found {len(injected_named_params)} attention parameter tensors.")
    
    # 4. Load cached dataset
    cached_splits = get_cached_tokenized_dataset("sst2", "distilbert", 128)
    train_loader = torch.utils.data.DataLoader(cached_splits["train"], batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(cached_splits["val"], batch_size=128, shuffle=False)
    
    # Zero-step validation
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
    acc_0 = accuracy_score(val_labels, val_preds) * 100.0
    f1_0 = f1_score(val_labels, val_preds, zero_division=0)
    print(f"Step 0 (Untrained): Acc={acc_0:.2f}%, F1={f1_0:.4f}, % Positive Preds={p_pos:.2f}% (Dataset actual positive=50.92%)")
    
    # 5. Training loop with per-step logging
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for step, (b_ids, b_mask, b_lbl) in enumerate(train_loader):
        if step >= steps:
            break
        optimizer.zero_grad()
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_lbl = b_lbl.to(device)
        
        out = model(input_ids=b_ids, attention_mask=b_mask, labels=b_lbl)
        loss = out.loss
        loss.backward()
        
        # Check gradient norms on head vs attention
        head_grads = []
        attn_grads = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                norm = p.grad.norm(2).item()
                if "classifier" in name:
                    head_grads.append(norm)
                elif "inner" in name or "attention" in name:
                    attn_grads.append(norm)
                    
        mean_head_grad = sum(head_grads) / max(1, len(head_grads))
        mean_attn_grad = sum(attn_grads) / max(1, len(attn_grads))
        
        optimizer.step()
        
        preds_train = torch.argmax(out.logits, dim=-1)
        p_train_pos = (preds_train == 1).float().mean().item() * 100.0
        
        if step in [0, 1, 4, 9, 14, 19]:
            print(f"Step {step:2d} | Loss: {loss.item():.4f} | Train Pos%: {p_train_pos:5.1f}% | Head Grad: {mean_head_grad:.5f} | Attn Grad: {mean_attn_grad:.5f}")
            
    # Final validation
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for b_ids, b_mask, b_lbl in val_loader:
            out = model(input_ids=b_ids.to(device), attention_mask=b_mask.to(device))
            logits = out.logits
            preds = torch.argmax(logits, dim=-1)
            val_preds.extend(preds.cpu().numpy().tolist())
            val_labels.extend(b_lbl.numpy().tolist())
            
    p_pos_final = sum(val_preds) / len(val_preds) * 100.0
    acc_final = accuracy_score(val_labels, val_preds) * 100.0
    f1_final = f1_score(val_labels, val_preds, zero_division=0)
    print(f"Final Step {steps}: Acc={acc_final:.2f}%, F1={f1_final:.4f}, % Positive Preds={p_pos_final:.2f}%")

if __name__ == "__main__":
    for mod in ["full", "mamba", "gated_delta_net", "linformer", "hgrn2_outer", "performer"]:
        try:
            diagnose_module(mod)
        except Exception as e:
            print(f"FAILED {mod}: {e}")
