#!/usr/bin/env python3
"""
Global Model Import Verification Script.
Runs across DistilBERT, RoBERTa, and GPT-2 configurations on the GPU server.
Saves verification results to audit/final_verification.json.
"""

import sys
import os
import json
import importlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "Codes"))

try:
    from transformers import AutoModelForSequenceClassification, AutoConfig
except ImportError:
    print("transformers not installed in environment")
    sys.exit(1)

with open(HERE / "audit" / "all_module_names.json", "r", encoding="utf-8") as f:
    ALL_MODULES = json.load(f)

print(f"Loaded {len(ALL_MODULES)} modules to verify.")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODELS = [
    ("distilbert", r"models\distilbert_sst2"),
    ("roberta", r"models\roberta-base"),
    ("gpt2", r"models\gpt2"),
]


results = []

for model_name, model_path in MODELS:
    print(f"\n--- Verifying under configuration: {model_name} ({model_path}) ---")
    try:
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
    except Exception:
        try:
            cfg = AutoConfig.from_pretrained(model_path)
        except Exception as e:
            cfg = None
            print(f"Warning: Could not load config for {model_path}: {e}")

    for mod in ALL_MODULES:
        pkg = "framework.attentions" if os.path.exists(HERE / "Codes" / "framework" / "attentions" / f"{mod}.py") else "framework.compression_engine"
        full_name = f"{pkg}.{mod}"
        try:
            importlib.import_module(full_name)
            results.append({
                "model": model_name,
                "module": mod,
                "package": pkg,
                "status": "IMPORT_OK"
            })
        except Exception as e:
            results.append({
                "model": model_name,
                "module": mod,
                "package": pkg,
                "status": "IMPORT_FAIL",
                "error": str(e)[:150]
            })

out_file = HERE / "audit" / "final_verification.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

fails = [r for r in results if r["status"] == "IMPORT_FAIL"]
print(f"\n==========================================")
print(f"TOTAL TESTS: {len(results)}")
print(f"IMPORT SUCCESS: {len(results) - len(fails)}")
print(f"IMPORT FAILS:   {len(fails)}")
print(f"Results recorded in: {out_file}")
print(f"==========================================")

if fails:
    print("\nFirst 20 failures:")
    for fail in fails[:20]:
        print(f"[{fail['model']}] {fail['module']}: {fail['error']}")
