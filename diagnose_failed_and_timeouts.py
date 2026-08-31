import sys
from pathlib import Path
sys.path.insert(0, "Codes")
sys.path.insert(0, ".")

import torch
import traceback
from transformers import AutoModelForSequenceClassification
from framework.eval_all_models_datasets import apply_module_to_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

failed_mods = ["liftr", "mla", "primal", "rala_compressor", "upi_stuffed_mamba", "vig", "vspec"]
timeout_mods = ["lsh", "mamba_svd_flash", "oats", "rotational_slicegpt", "rpca", "samba_m3", "samba_moe", "simple_ssm", "svd", "svd-mamba", "unified_rope", "uniql"]

print("=================== DIAGNOSING 7 FAILED MODULES ===================")
for mod in failed_mods:
    try:
        model = AutoModelForSequenceClassification.from_pretrained("D:/SVD_Project/models/distilbert_sst2")
        mod_type = "compressor" if "compressor" in mod else "attention"
        model, _ = apply_module_to_model(model, mod, mod_type, rank=16, device=device)
        model = model.to(device)
        inp = torch.randint(0, 1000, (2, 128), device=device)
        mask = torch.ones((2, 128), device=device)
        out = model(input_ids=inp, attention_mask=mask)
        print(f"PASS: {mod:25s} | Output shape: {out.logits.shape}")
    except Exception as e:
        print(f"FAIL: {mod:25s} | {e}")

print("\n=================== DIAGNOSING 12 TIMEOUT MODULES ===================")
for mod in timeout_mods:
    try:
        model = AutoModelForSequenceClassification.from_pretrained("D:/SVD_Project/models/distilbert_sst2")
        mod_type = "compressor" if ("slicegpt" in mod or "rpca" in mod) else "attention"
        model, _ = apply_module_to_model(model, mod, mod_type, rank=16, device=device)
        model = model.to(device)
        inp = torch.randint(0, 1000, (2, 128), device=device)
        mask = torch.ones((2, 128), device=device)
        
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        out = model(input_ids=inp, attention_mask=mask)
        t1.record()
        torch.cuda.synchronize()
        lat = t0.elapsed_time(t1)
        print(f"PASS: {mod:25s} | Forward latency = {lat:.2f} ms")
    except Exception as e:
        print(f"FAIL: {mod:25s} | {e}")
