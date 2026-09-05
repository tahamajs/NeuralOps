import torch
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
sys.path.insert(0, "Codes")
from framework.run_quality_sweep import finetune, finetune_lora, prepare_eval_batches, evaluate_comprehensive

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained("models/roberta-base")
ds = load_from_disk("Codes/framework/data/glue_data/sst2")
train, val = ds["train"], ds["validation"]

# Method 1: Head fine-tune (200 steps)
m1 = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(dev)
print("Running 200 step head fine-tune on roberta-base...", flush=True)
finetune(m1, tok, train, "sst2", 200, 128, 8, dev, True)
vb = prepare_eval_batches(tok, val, "sst2", 128, 8, dev)
res1 = evaluate_comprehensive(m1, tok, val, "sst2", 128, 8, dev, precomputed_batches=vb)
print(f"Head-finetuned roberta acc: {res1['accuracy_pct']:.2f}%", flush=True)

# Method 2: LoRA fine-tune (200 steps)
from framework.utils.lora import apply_lora, find_all_linear_submodule_names
m2 = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(dev)
dyn_targets = find_all_linear_submodule_names(m2)
m2, num_l = apply_lora(m2, r=16, alpha=32, dropout=0.05, target_modules=dyn_targets)
m2.to(dev)
head = next((getattr(m2, attr, None) for attr in ("score", "classifier", "classification_head") if getattr(m2, attr, None) is not None), None)
if head is not None:
    for p in head.parameters(): p.requires_grad = True
print("Running 200 step LoRA fine-tune on roberta-base...", flush=True)
finetune_lora(m2, tok, train, "sst2", 200, 128, 8, dev, True, lr=2e-4)
res2 = evaluate_comprehensive(m2, tok, val, "sst2", 128, 8, dev, precomputed_batches=vb)
print(f"LoRA-finetuned roberta acc: {res2['accuracy_pct']:.2f}%", flush=True)

if res2["accuracy_pct"] >= 75.0:
    torch.save(m2.state_dict(), "results/roberta_sst2_lora_baseline/finetuned_model.pt")
    print(f"Saved optimal LoRA checkpoint ({res2['accuracy_pct']:.2f}%) to results/roberta_sst2_lora_baseline/finetuned_model.pt", flush=True)
elif res1["accuracy_pct"] >= 70.0:
    torch.save(m1.state_dict(), "results/roberta_sst2_lora_baseline/finetuned_model.pt")
    print(f"Saved optimal Head checkpoint ({res1['accuracy_pct']:.2f}%) to results/roberta_sst2_lora_baseline/finetuned_model.pt", flush=True)
