import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, "Codes")
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from framework.eval_all_models_datasets import apply_module_to_model
from framework.data.cached_tensors import get_cached_tokenized_dataset
from torch.optim import AdamW

def build_discriminative_optimizer(model, base_lr=3e-5, head_lr=1e-4, new_attn_lr=3e-4, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight", "ln.weight"]
    head_decay, head_no_decay = [], []
    injected_decay, injected_no_decay = [], []
    backbone_decay, backbone_no_decay = [], []
    head_kw = ["classifier", "pre_classifier", "score", "lm_head"]
    injected_kw = ["attention", "compressor", "linear_attn", "mamba", "ssm", "moe", "lora", "svd", "conv", "gate", "kernel", "proj", "inner"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_nd = any(nd in name for nd in no_decay)
        is_head = any(hk in name for hk in head_kw)
        is_inj = not is_head and any(ik in name for ik in injected_kw)
        if is_head:
            (head_no_decay if is_nd else head_decay).append(param)
        elif is_inj:
            (injected_no_decay if is_nd else injected_decay).append(param)
        else:
            (backbone_no_decay if is_nd else backbone_decay).append(param)

    groups = []
    if head_decay: groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay})
    if head_no_decay: groups.append({"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0})
    if injected_decay: groups.append({"params": injected_decay, "lr": new_attn_lr, "weight_decay": weight_decay})
    if injected_no_decay: groups.append({"params": injected_no_decay, "lr": new_attn_lr, "weight_decay": 0.0})
    if backbone_decay: groups.append({"params": backbone_decay, "lr": base_lr, "weight_decay": weight_decay})
    if backbone_no_decay: groups.append({"params": backbone_no_decay, "lr": base_lr, "weight_decay": 0.0})
    return AdamW(groups)

test_modules = ["mamba", "gated_delta_net", "linformer", "hgrn2_outer", "performer"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cached = get_cached_tokenized_dataset("sst2", "distilbert", 128)
loader = torch.utils.data.DataLoader(cached["train"], batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(cached["val"], batch_size=128, shuffle=False)

print("=== PHASE 1 UNIT TEST: GRADIENT FLOW & PREDICTION DIVERSITY (20 STEPS) ===")
for mod_name in test_modules:
    model = AutoModelForSequenceClassification.from_pretrained("D:/SVD_Project/models/distilbert_sst2")
    mod_type = "compressor" if "compressor" in mod_name else "attention"
    model, _ = apply_module_to_model(model, mod_name, mod_type, rank=16, device=device)
    model = model.to(device)
    optimizer = build_discriminative_optimizer(model)
    
    # Pre-flight gradient check
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    for n, p in model.named_parameters():
        assert p.requires_grad, f"Frozen parameter: {n}"
        assert id(p) in opt_ids, f"Missing from optimizer: {n}"
    
    model.train()
    step = 0
    for b_idx, (b_ids, b_mask, b_lbl) in enumerate(loader):
        if step >= 20: break
        inp = {"input_ids": b_ids.to(device), "attention_mask": b_mask.to(device), "labels": b_lbl.to(device)}
        optimizer.zero_grad()
        out = model(**inp)
        loss = out.loss
        loss.backward()
        
        # Measure grad norms
        head_grad = torch.norm(torch.stack([torch.norm(p.grad.detach()) for n, p in model.named_parameters() if any(h in n for h in ["classifier", "pre_classifier"]) and p.grad is not None]))
        inj_grad = torch.norm(torch.stack([torch.norm(p.grad.detach()) for n, p in model.named_parameters() if "attention" in n and p.grad is not None]))
        
        optimizer.step()
        step += 1
        if step in (1, 10, 20):
            preds = out.logits.argmax(dim=-1)
            pos_ratio = (preds == 1).float().mean().item() * 100
            print(f"[{mod_name:18s}] Step {step:2d}/20 | Loss: {loss.item():.4f} | HeadGradNorm: {head_grad.item():.4f} | InjGradNorm: {inj_grad.item():.4f} | PosPred%: {pos_ratio:.1f}%")

    # Eval accuracy
    model.eval()
    corr, tot = 0, 0
    with torch.no_grad():
        for v_ids, v_mask, v_lbl in val_loader:
            v_out = model(input_ids=v_ids.to(device), attention_mask=v_mask.to(device))
            corr += (v_out.logits.argmax(dim=-1).cpu() == v_lbl).sum().item()
            tot += len(v_lbl)
    acc = corr / tot * 100
    print(f"--> [{mod_name:18s}] 20-Step Accuracy: {acc:.2f}% (Previously collapsed to 50.92% / 49.08%)")
    print("-" * 80)
