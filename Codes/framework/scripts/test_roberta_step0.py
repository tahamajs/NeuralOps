import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from framework.core.attention_utils import inject_attention
from framework.run_quality_sweep import evaluate_comprehensive, prepare_eval_batches
from datasets import load_from_disk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained("models/roberta-base").to(device)
ckpt_path = "results/unified_asvd/roberta/finetuned_model.pt"
ckpt = torch.load(ckpt_path, map_location=device)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
elif "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
else:
    model.load_state_dict(ckpt)

tokenizer = AutoTokenizer.from_pretrained("models/roberta-base")
val_ds = load_from_disk("Codes/framework/data/glue_data/sst2")["validation"]
batches = prepare_eval_batches(tokenizer, val_ds, "sst2", 128, 8, device)

perf_base = evaluate_comprehensive(model, tokenizer, val_ds, "sst2", 128, 8, device, fp16=True, precomputed_batches=batches)
print(f"Pre-inject baseline accuracy: {perf_base['accuracy_pct']:.2f}%")

inject_attention(model, "adaptive_span", rank=32)
model.to(device)
perf_step0 = evaluate_comprehensive(model, tokenizer, val_ds, "sst2", 128, 8, device, fp16=True, precomputed_batches=batches)
print(f"Post-inject Step 0 accuracy: {perf_step0['accuracy_pct']:.2f}%")
