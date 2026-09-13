import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.core.attention_utils import inject_attention
from framework.run_quality_sweep import evaluate, finetune_lora, prepare_train_batches
from framework.utils.compat import get_recovery_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ds = load_from_disk("Codes/framework/data/glue_data/sst2")
val = ds["validation"]
train = ds["train"]
tok = AutoTokenizer.from_pretrained("models/roberta-base")
tb = prepare_train_batches(tok, train, "sst2", 128, 8, device, max_batches=50)

# Test A: m in FP16 before injection (like run_quality_sweep)
mA = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(device)
ckpt = torch.load("results/unified_asvd/roberta/finetuned_model.pt", map_location=device)
mA.load_state_dict(ckpt, strict=False)
for p in mA.parameters():
    p.data = p.data.half()
inject_attention(mA, "adaptive_span", rank=16)
trainableA = get_recovery_params(mA)
finetune_lora(mA, tok, train, "sst2", 50, 128, 8, device, fp16=True, lr=2e-4, precomputed_batches=tb)
accA = evaluate(mA, tok, val, "sst2", 128, 32, device, fp16=True)
print("Test A (FP16 base):", accA)

# Test B: m in FP32 before injection (like debug_difference)
mB = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(device)
mB.load_state_dict(ckpt, strict=False)
inject_attention(mB, "adaptive_span", rank=16)
trainableB = get_recovery_params(mB)
finetune_lora(mB, tok, train, "sst2", 50, 128, 8, device, fp16=True, lr=2e-4, precomputed_batches=tb)
accB = evaluate(mB, tok, val, "sst2", 128, 32, device, fp16=True)
print("Test B (FP32 base):", accB)
