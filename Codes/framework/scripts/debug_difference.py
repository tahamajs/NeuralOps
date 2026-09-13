import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.core.attention_utils import inject_attention
from framework.run_quality_sweep import evaluate, finetune_lora, prepare_train_batches, prepare_eval_batches
from framework.utils.compat import get_recovery_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ds = load_from_disk("Codes/framework/data/glue_data/sst2")
val = ds["validation"]
train = ds["train"]
tok = AutoTokenizer.from_pretrained("models/roberta-base")

# Let's compare:
# Run 1: Exactly like sweep_roberta_lrs.py
m1 = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(device)
ckpt = torch.load("results/unified_asvd/roberta/finetuned_model.pt", map_location=device)
m1.load_state_dict(ckpt, strict=False)
inject_attention(m1, "adaptive_span", rank=16)

for p in m1.parameters(): p.requires_grad = False
for n, p in m1.named_parameters():
    if "lora_" in n or "span_logits" in n or "classifier.out_proj" in n:
        p.requires_grad = True

tb1 = prepare_train_batches(tok, train, "sst2", 128, 8, device, max_batches=50)
finetune_lora(m1, tok, train, "sst2", 50, 128, 8, device, fp16=True, lr=2e-4, precomputed_batches=tb1)
acc1 = evaluate(m1, tok, val, "sst2", 128, 32, device, fp16=True)
print("Run 1 (sweep script approach):", acc1)

# Run 2: Using get_recovery_params
m2 = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(device)
m2.load_state_dict(ckpt, strict=False)
inject_attention(m2, "adaptive_span", rank=16)

trainable = get_recovery_params(m2)
tb2 = prepare_train_batches(tok, train, "sst2", 128, 8, device, max_batches=50)
finetune_lora(m2, tok, train, "sst2", 50, 128, 8, device, fp16=True, lr=2e-4, precomputed_batches=tb2)
acc2 = evaluate(m2, tok, val, "sst2", 128, 32, device, fp16=True)
print("Run 2 (get_recovery_params approach):", acc2)

# Check differences between m1 and m2
for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
    if p1.requires_grad != p2.requires_grad:
        print(f"requires_grad mismatch: {n1}: m1={p1.requires_grad}, m2={p2.requires_grad}")
