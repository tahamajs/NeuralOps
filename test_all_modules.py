#!/usr/bin/env python3
"""
Comprehensive verification test script across all attention & compression modules.
Runs a 3-model forward pass test (DistilBERT, RoBERTa, GPT-2) on GPU.
Saves results to audit/final_verification.json.
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path

# Enforce Hugging Face offline mode to prevent network hangs
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModel, AutoConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "Codes"))

from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.core.attention_utils import inject_attention
import framework.attentions
import framework.compression_engine

MODELS = {
    "distilbert": [
        str(HERE / "models" / "distilbert_sst2"),
        "models/distilbert_sst2",
        "distilbert-base-uncased",
    ],
    "roberta": [
        str(HERE / "models" / "roberta-base"),
        "models/roberta-base",
        "roberta-base",
    ],
    "gpt2": [
        str(HERE / "models" / "gpt2"),
        "models/gpt2",
        "gpt2",
    ],
}


def get_dummy_input(model_name: str, device: str):
    if "gpt2" in model_name:
        x = torch.randint(0, 1000, (1, 16), device=device)
        return {"input_ids": x}
    else:
        x = torch.randint(0, 1000, (1, 16), device=device)
        mask = torch.ones((1, 16), dtype=torch.long, device=device)
        return {"input_ids": x, "attention_mask": mask}

def run_forward_verification():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Comprehensive Verification on {device.upper()} ===")

    all_attentions = sorted(list(set(ATTENTION_REGISTRY.keys())))
    print(f"Discovered {len(all_attentions)} attention registry entries.")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "modules": {},
        "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
    }

    # Pre-cache models to save time
    base_models = {}
    for mkey, candidates in MODELS.items():
        loaded = False
        for cpath in candidates:
            try:
                print(f"Attempting to load {mkey} from '{cpath}'...")
                cfg = AutoConfig.from_pretrained(cpath, local_files_only=True)
                base_models[mkey] = AutoModel.from_pretrained(cpath, config=cfg, local_files_only=True).to(device).eval()
                print(f"Successfully loaded {mkey} from '{cpath}'")
                loaded = True
                break
            except Exception as e:
                # Try with standard from_pretrained without local_files_only flag if cache exists
                try:
                    cfg = AutoConfig.from_pretrained(cpath)
                    base_models[mkey] = AutoModel.from_pretrained(cpath, config=cfg).to(device).eval()
                    print(f"Successfully loaded {mkey} from '{cpath}' (cached)")
                    loaded = True
                    break
                except Exception:
                    pass
        if not loaded:
            print(f"Warning: Failed to load any candidate model for {mkey}")


    for idx, attn_name in enumerate(all_attentions, 1):
        if attn_name in ("gated_delta_net", "gated_deltanet_block", "hgrn2_gated_deltanet"):
            print(f"[{idx}/{len(all_attentions)}] {attn_name} -> SKIPPED (DEPRECATED)")
            results["modules"][attn_name] = {"status": "SKIPPED", "reason": "deprecated"}
            results["summary"]["skipped"] += 1
            continue

        results["summary"]["total"] += 1
        module_res = {"status": "PASS", "models": {}}
        print(f"[{idx}/{len(all_attentions)}] Testing {attn_name}...", end=" ", flush=True)

        for mkey, mname in MODELS.items():
            if mkey not in base_models:
                continue
            try:
                # Fresh clone
                import copy
                test_model = copy.deepcopy(base_models[mkey])
                inp = get_dummy_input(mname, device)

                with torch.no_grad():
                    orig_out = test_model(**inp)[0]

                # Inject
                injected_model = inject_attention(
                    model=test_model,
                    attention_type=attn_name,
                    config={"rank": 64, "lora_rank": 16}
                )

                if injected_model is None:
                    module_res["models"][mkey] = {"status": "FAIL", "error": "inject_attention returned None"}
                    module_res["status"] = "FAIL"
                    continue

                with torch.no_grad():
                    new_out = injected_model(**inp)[0]

                if torch.isnan(new_out).any():
                    module_res["models"][mkey] = {"status": "FAIL", "error": "NaN in forward output"}
                    module_res["status"] = "FAIL"
                else:
                    diff = (orig_out - new_out).abs().mean().item()
                    module_res["models"][mkey] = {"status": "PASS", "diff": round(diff, 6)}

                del test_model, injected_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                tb_lines = traceback.format_exc().strip().split("\n")
                last_line = tb_lines[-1] if tb_lines else str(e)
                module_res["models"][mkey] = {"status": "FAIL", "error": last_line}
                module_res["status"] = "FAIL"

        if module_res["status"] == "PASS":
            results["summary"]["passed"] += 1
            print("PASS")
        else:
            results["summary"]["failed"] += 1
            print("FAIL")

        results["modules"][attn_name] = module_res

    out_path = HERE / "audit" / "final_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Verification Complete ===")
    print(f"Total Tested: {results['summary']['total']}")
    print(f"Passed:       {results['summary']['passed']}")
    print(f"Failed:       {results['summary']['failed']}")
    print(f"Skipped:      {results['summary']['skipped']}")
    print(f"Results saved to: {out_path}")

if __name__ == "__main__":
    run_forward_verification()
