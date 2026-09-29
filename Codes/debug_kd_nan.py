#!/usr/bin/env python3
import copy
import math
import sys
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Codes"))

from framework.eval_all_models_datasets import apply_module_to_model
from kd_quick_test import set_train_scope, compute_kl_divergence, build_train_chunks

print("Loading dataset and tokenizer...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
tok = AutoTokenizer.from_pretrained("models/gpt2_wikitext")
chunks = build_train_chunks(ds, tok, max_train_texts=50, seq_len=256).cuda()

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("models/gpt2_wikitext").cuda().half()
student = copy.deepcopy(model)
student, _ = apply_module_to_model(student, "triton_svd_linear_attention", "attention", rank=64, device="cuda")
student.half()
set_train_scope(student, "mamba")

# Keep frozen backbone in half, but ensure trainable parameters are float32 (master weights)
for p in student.parameters():
    if p.requires_grad:
        p.data = p.data.float()

trainable = [p for p in student.parameters() if p.requires_grad]
print("Trainable dtypes:", set(p.dtype for p in trainable))
opt = torch.optim.AdamW(trainable, lr=2.5e-5, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler(enabled=True)

batch = chunks[:4]
print("Running Step 1 forward...")
with torch.cuda.amp.autocast(dtype=torch.float16):
    teacher_out = model(batch)
    student_out = student(batch)
    loss = compute_kl_divergence(student_out.logits, teacher_out.logits, temperature=4.0)

print(f"Initial loss: {loss.item()}")
scaler.scale(loss).backward()
scaler.unscale_(opt)

print("--- GRAD STATS BEFORE CLIP ---")
for name, p in student.named_parameters():
    if p.requires_grad:
        if p.grad is None:
            print(f"GRAD IS NONE: {name}")
        else:
            has_nan = torch.isnan(p.grad).any().item()
            has_inf = torch.isinf(p.grad).any().item()
            gnorm = p.grad.norm().item()
            print(f"{name}: grad_shape={list(p.grad.shape)}, norm={gnorm}, nan={has_nan}, inf={has_inf}")

print(f"Total trainable params: {len(trainable)}")
print(f"Params with grad: {sum(1 for p in trainable if p.grad is not None)}")

total_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=0.5)
print(f"Total grad norm before clip: {total_norm}")
print("Checking grads after clip_grad_norm_:")
for name, p in student.named_parameters():
    if p.requires_grad and p.grad is not None:
        if torch.isnan(p.grad).any().item() or torch.isinf(p.grad).any().item():
            print(f"AFTER CLIP NAN/INF: {name}")

scaler.step(opt)
scaler.update()
opt.zero_grad()

print("--- PARAM STATS AFTER STEP ---")
any_nan = False
for name, p in student.named_parameters():
    if p.requires_grad:
        pnorm = p.norm().item()
        has_nan = torch.isnan(p).any().item()
        has_inf = torch.isinf(p).any().item()
        if has_nan or has_inf or pnorm > 1000:
            print(f"{name}: norm={pnorm:.4f}, nan={has_nan}, inf={has_inf}")
            any_nan = True
print(f"Any params nan after step?: {any_nan}")

print("Running Step 2 forward...")
with torch.cuda.amp.autocast(dtype=torch.float16):
    out2 = student(batch)
    print("out2.logits has nan?:", torch.isnan(out2.logits).any().item())
    print("out2.logits has inf?:", torch.isinf(out2.logits).any().item())
    if torch.isnan(out2.logits).any().item():
        # Check layer by layer where NaNs first appeared
        pass
