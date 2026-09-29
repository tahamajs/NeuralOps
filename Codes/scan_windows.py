#!/usr/bin/env python3
import copy
import sys
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Codes"))

from framework.eval_all_models_datasets import apply_module_to_model

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
tok = AutoTokenizer.from_pretrained("models/gpt2_wikitext")
model = AutoModelForCausalLM.from_pretrained("models/gpt2_wikitext").cuda().half()
student, _ = apply_module_to_model(copy.deepcopy(model), "triton_svd_linear_attention", "attention", rank=64, device="cuda")
student.eval()

text = "\n\n".join(ds["text"])
enc = tok(text, return_tensors="pt")
ids_all = enc.input_ids[:, :10000]

print("Scanning sliding windows from 0 to 10000...")
nan_count = 0
for b in range(0, 10000, 128):
    e = min(b + 256, 10000)
    ids = ids_all[:, b:e].cuda()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = student(ids, labels=ids)
            l = out.loss.item()
            if torch.isnan(out.loss) or torch.isinf(out.loss):
                nan_count += 1
                print(f"NaN at window [{b}:{e}]")
            elif b % 1024 == 0:
                print(f"Window [{b}:{e}] loss: {l:.4f}")

print(f"Total NaN windows: {nan_count}")
