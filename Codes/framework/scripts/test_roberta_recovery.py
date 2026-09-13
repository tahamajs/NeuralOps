import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.core.attention_utils import inject_attention
from framework.run_quality_sweep import evaluate, finetune_lora, prepare_train_batches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ds = load_from_disk("Codes/framework/data/glue_data/sst2")
val = ds["validation"]
train = ds["train"]
tok = AutoTokenizer.from_pretrained("models/roberta-base")

# Test 1: Baseline
m = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(device)
ckpt = torch.load("results/unified_asvd/roberta/finetuned_model.pt", map_location=device)
m.load_state_dict(ckpt, strict=False)

inject_attention(m, "adaptive_span", rank=16)
m.eval()
acc_step0 = evaluate(m, tok, val, "sst2", 128, 32, device, fp16=True)
print("Injected Step 0 accuracy:", acc_step0)

# Freeze everything EXCEPT LoRA and span_logits
for p in m.parameters():
    p.requires_grad = False

for n, p in m.named_parameters():
    if "lora_" in n or "span_logits" in n:
        p.requires_grad = True

trainable_names = [n for n, p in m.named_parameters() if p.requires_grad]
print("Trainable parameter tensors count:", len(trainable_names))
print("Trainable parameters total:", sum(p.numel() for p in m.parameters() if p.requires_grad))
print("Sample trainable names:", trainable_names[:5], "...", trainable_names[-3:])

tb = prepare_train_batches(tok, train, "sst2", 128, 8, device, max_batches=50)

finetune_lora(m, tok, train, "sst2", 50, 128, 8, device, fp16=True, lr=5e-5, precomputed_batches=tb)

acc_after = evaluate(m, tok, val, "sst2", 128, 32, device, fp16=True)
print("Accuracy after 50 recovery steps (classifier frozen):", acc_after)

# Print span values
spans = [p.data.tolist() for n, p in m.named_parameters() if "span_logits" in n]
print("Final span logits layer 0:", spans[0] if spans else "None")
