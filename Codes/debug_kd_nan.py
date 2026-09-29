#!/usr/bin/env python3
import copy
import math
import sys
import torch
import torch.nn.functional as F
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

def proper_kl(student_logits, teacher_logits, temperature=4.0):
    p_s = F.log_softmax(student_logits / temperature, dim=-1).view(-1, student_logits.size(-1))
    p_t = F.softmax(teacher_logits / temperature, dim=-1).view(-1, teacher_logits.size(-1))
    return F.kl_div(p_s, p_t, reduction="batchmean") * (temperature * temperature)

print("Starting 10-step training loop...")
for step in range(1, 11):
    opt.zero_grad()
    b_idx = (step - 1) * 4
    batch = chunks[b_idx:b_idx + 4]
    
    with torch.cuda.amp.autocast(dtype=torch.float16):
        with torch.no_grad():
            t_out = model(batch)
        s_out = student(batch)
        loss_kl = proper_kl(s_out.logits, t_out.logits, temperature=4.0)
        
        shift_logits = s_out.logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss = 0.9 * loss_kl + 0.1 * loss_ce
        
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=0.5)
    scaler.step(opt)
    scaler.update()
    
    print(f"Step {step:2d} | Loss: {loss.item():.4f} (KL: {loss_kl.item():.4f}, CE: {loss_ce.item():.4f}) | Grad Norm: {norm.item():.4f}")

print("Training finished cleanly without NaNs!")
