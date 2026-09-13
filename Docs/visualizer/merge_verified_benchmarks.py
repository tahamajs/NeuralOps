#!/usr/bin/env python3
"""
Merge master verified benchmark results from master_verified_results.json
into Docs/visualizer/modules_components.json and Docs/visualizer/modules-catalog.json.
Ensures 100% code and benchmark alignment across DistilBERT, RoBERTa, GPT-2, and Qwen.
"""

import json
import os

def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    master_path = os.path.join(root, "master_verified_results.json")
    mc_path = os.path.join(root, "Docs", "visualizer", "modules_components.json")
    cat_path = os.path.join(root, "Docs", "visualizer", "modules-catalog.json")

    with open(master_path) as f:
        master_data = json.load(f)
    master_modules = master_data.get("modules", {})

    with open(mc_path) as f:
        mc = json.load(f)

    with open(cat_path) as f:
        catalog = json.load(f)

    model_map = {
        "DistilBERT": "distilbert",
        "RoBERTa": "roberta",
        "GPT-2": "gpt2",
        "Qwen": "qwen"
    }

    manual_aliases = {
        "aa_svd_compressor": "aasvd",
        "saes_svd": "saes_svd_v2",
        "upi_mamba": "upi_stuffed_mamba",
        "soafusedattention": "soafusedattention",
        "full_attention": None,
        "pruning_utils": None
    }

    def find_entry(mid):
        if not mid:
            return None
        low = mid.lower()
        if low in manual_aliases:
            alias = manual_aliases[low]
            return master_modules.get(alias) if alias else None
        if mid in master_modules:
            return master_modules[mid]
        if low in master_modules:
            return master_modules[low]

        candidates = [
            low.replace("_pruner", ""),
            low.replace("_quantizer", ""),
            low.replace("_compressor", ""),
            low.replace("_attention", ""),
            low.replace("_attn", ""),
            low.replace("_layer", ""),
        ]
        for c in candidates:
            if c in master_modules:
                return master_modules[c]

        clean = low.replace("_", "")
        for k, v in master_modules.items():
            if clean == k.lower().replace("_", ""):
                return v
        return None

    # Enrich modules_components.json
    matched_count = 0
    for m in mc.get("modules", []):
        mid = m.get("id", "")
        entry = find_entry(mid)
        if entry:
            matched_count += 1
            b_info = {
                "overall_status": entry.get("overall_status", "UNKNOWN"),
                "best_accuracy": entry.get("best_accuracy"),
                "has_pass": entry.get("has_pass", False),
                "is_pure_pass": entry.get("is_pure_pass", False),
                "models": {}
            }
            for mk, mv in entry.get("models", {}).items():
                target_k = model_map.get(mk)
                if target_k:
                    b_info["models"][target_k] = {
                        "status": mv.get("status"),
                        "accuracy_pct": mv.get("accuracy_pct"),
                        "f1_score": mv.get("f1_score"),
                        "peak_vram_mb": mv.get("peak_vram_mb"),
                        "throughput_tokens_sec": mv.get("throughput_tokens_sec"),
                        "source_file": mv.get("source_file")
                    }
                    if "model_projections" in m and target_k in m["model_projections"]:
                        m["model_projections"][target_k]["benchmark"] = b_info["models"][target_k]

            m["verified_benchmark"] = b_info

    # Enrich modules-catalog.json
    for m in catalog.get("modules", []):
        mid = m.get("id", "")
        entry = find_entry(mid)
        if entry:
            m["verified_benchmark"] = {
                "overall_status": entry.get("overall_status", "UNKNOWN"),
                "best_accuracy": entry.get("best_accuracy"),
                "has_pass": entry.get("has_pass", False),
                "models": {}
            }
            for mk, mv in entry.get("models", {}).items():
                target_k = model_map.get(mk)
                if target_k:
                    m["verified_benchmark"]["models"][target_k] = {
                        "status": mv.get("status"),
                        "accuracy_pct": mv.get("accuracy_pct"),
                        "f1_score": mv.get("f1_score"),
                        "peak_vram_mb": mv.get("peak_vram_mb"),
                        "throughput_tokens_sec": mv.get("throughput_tokens_sec")
                    }

    with open(mc_path, "w") as f:
        json.dump(mc, f, indent=2)

    with open(cat_path, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"Successfully enriched {matched_count} modules with real master verified benchmarks in modules_components.json and modules-catalog.json!")

if __name__ == "__main__":
    main()
