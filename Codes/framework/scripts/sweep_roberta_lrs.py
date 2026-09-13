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
tb = prepare_train_batches(tok, train, "sst2", 128, 8, device, max_batches=50)

# Mode A: classifier completely frozen (only lora and span_logits)
# Mode B: classifier.dense frozen, classifier.out_proj trainable
for mode, lr in [("frozen_all", 1e-5), ("frozen_all", 2e-4), ("freeze_dense_only", 1e-5), ("freeze_dense_only", 2e-4)]:
    m = AutoModelForSequenceClassification.from_pretrained("models/roberta-base", num_labels=2).to(device)
    ckpt = torch.load("results/unified_asvd/roberta/finetuned_model.pt", map_location=device)
    m.load_state_dict(ckpt, strict=False)
    inject_attention(m, "adaptive_span", rank=16)

    for p in m.parameters():
        p.requires_grad = False

    for n, p in m.named_parameters():
        if "lora_" in n or "span_logits" in n:
            p.requires_grad = True
        if mode == "freeze_dense_only" and "classifier.out_proj" in n:
            p.requires_grad = True

    finetune_lora(m, tok, train, "sst2", 50, 128, 8, device, fp16=True, lr=lr, precomputed_batches=tb)
    acc = evaluate(m, tok, val, "sst2", 128, 32, device, fp16=True)
    print(f"Mode={mode}, lr={lr:.1e} -> Val Acc: {acc:.2f}%")
