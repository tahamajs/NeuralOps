import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path
sys.path.insert(0, "Codes")
sys.path.insert(0, ".")

import torch
import time
import traceback
from transformers import AutoModelForSequenceClassification
from framework.eval_all_models_datasets import apply_module_to_model
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_mods = [
    # 7 failed
    "liftr", "mla", "primal", "rala_compressor", "upi_stuffed_mamba", "vig", "vspec",
    # 12 timeouts
    "lsh", "mamba_svd_flash", "oats", "rotational_slicegpt", "rpca", "samba_m3", "samba_moe", "simple_ssm", "svd", "svd-mamba", "unified_rope", "uniql"
]

print("=== VERIFYING CATEGORY & 5-STEP TRAINING FOR 19 TARGET MODULES ===")
for mod in test_mods:
    mod_type = "compressor" if (mod in COMPRESSOR_REGISTRY or "compressor" in mod or "slicegpt" in mod or "svd" in mod and mod in COMPRESSOR_REGISTRY) else "attention"
    try:
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained("D:/SVD_Project/models/distilbert_sst2")
        model, _ = apply_module_to_model(model, mod, mod_type, rank=16, device=device)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Run 5 training steps
        losses = []
        for step in range(5):
            inp = torch.randint(0, 1000, (16, 128), device=device)
            mask = torch.ones((16, 128), device=device)
            labels = torch.randint(0, 2, (16,), device=device)
            out = model(input_ids=inp, attention_mask=mask, labels=labels)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        dur = time.time() - t0
        print(f"[PASS] [{mod_type:10s}] {mod:25s} | 5 steps: {dur:.2f}s | Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    except Exception as e:
        print(f"[FAIL] [{mod_type:10s}] {mod:25s} | FAILED: {e}")
        lines = traceback.format_exc().strip().splitlines()
        print("   " + lines[-2] if len(lines) > 1 else "")
        print("   " + lines[-1])
