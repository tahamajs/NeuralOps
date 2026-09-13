import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.core.attention_utils import inject_attention
from framework.run_quality_sweep import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ds = load_from_disk("Codes/framework/data/glue_data/sst2")
val = ds["validation"]
tok = AutoTokenizer.from_pretrained("models/roberta-base")

m1 = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(device)
ckpt = torch.load("results/unified_asvd/roberta/finetuned_model.pt", map_location=device)
m1.load_state_dict(ckpt, strict=False)
m1.eval()

m2 = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(device)
m2.load_state_dict(ckpt, strict=False)
inject_attention(m2, "adaptive_span", rank=16)
m2.eval()

# Check sample logits with max_length padding
enc = tok(val[:4]["sentence"], padding="max_length", max_length=128, truncation=True, return_tensors="pt").to(device)
with torch.no_grad():
    out1 = m1(**enc).logits
    out2 = m2(**enc).logits

print("m1 logits:\n", out1)
print("m2 logits:\n", out2)
print("Logits max abs diff:", (out1 - out2).abs().max().item())

acc1 = evaluate(m1, tok, val, "sst2", 128, 32, device, fp16=True)
print("m1 baseline validation accuracy:", acc1)

acc2 = evaluate(m2, tok, val, "sst2", 128, 32, device, fp16=True)
print("m2 injected step 0 validation accuracy:", acc2)
