"""Normalize result JSON files to canonical schema."""
import csv
import json
import re
from pathlib import Path

CANONICAL_FIELDS = [
    "source_file", "module", "model", "dataset", "rank",
    "accuracy", "f1", "loss", "perplexity",
    "peak_vram_mb", "latency_ms", "duration_s",
    "compression_ratio", "status"
]

def infer_metadata_from_path(path: Path) -> dict:
    """Infer module/model/dataset from file path."""
    meta = {"module": "", "model": "", "dataset": "", "rank": ""}
    
    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    import json as _json
    
    aliases = _json.load(open("neuralops/core/deprecated_aliases.json"))
    canonical_keys = set(ATTENTION_REGISTRY.keys()) | set(COMPRESSOR_REGISTRY.keys())
    valid = canonical_keys | set(aliases.keys())
    
    parts = list(path.parts)
    # Check whole parts first in reverse order
    for part in reversed(parts):
        part_clean = part.replace("_full_metrics", "").replace(".json", "")
        if part_clean in valid:
            meta["module"] = aliases.get(part_clean, part_clean)
            break
        # Pattern match known models & datasets
        for m in ["distilbert", "roberta", "gpt2", "llama", "qwen"]:
            if m in part_clean.lower() and not meta["model"]:
                meta["model"] = m
        for ds in ["sst2", "mnli", "ag_news", "agnews", "wikitext2", "mrpc", "qqp", "c4", "qnli"]:
            if ds in part_clean.lower() and not meta["dataset"]:
                meta["dataset"] = ds

    # If module still empty, split tokens
    if not meta["module"]:
        for part in reversed(parts):
            tokens = re.split(r"[_\-\.]", part.replace(".json", ""))
            # Check multi-token candidate sub-strings
            joined = "_".join(tokens)
            for v in sorted(valid, key=len, reverse=True):
                if v in joined:
                    meta["module"] = aliases.get(v, v)
                    break
            if meta["module"]:
                break

    # Look for rank in tokens (e.g. rank_64, r64)
    for part in parts:
        m_rank = re.search(r"(?:rank[_-]?|r)(\d+)", part.lower())
        if m_rank:
            meta["rank"] = m_rank.group(1)

    return meta

def normalize_file(path: Path) -> dict:
    """Normalize a single JSON file to canonical schema."""
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    
    meta = infer_metadata_from_path(path)
    
    canonical = {
        "source_file": str(path),
        "module": meta["module"],
        "model": meta["model"],
        "dataset": meta["dataset"],
        "rank": meta["rank"],
        "metrics": {},
        "resources": {},
        "status": "PASS",
    }
    
    if isinstance(data, dict):
        # Override metadata from explicit top-level dict fields if present
        if data.get("module"): canonical["module"] = data["module"]
        if data.get("model"): canonical["model"] = data["model"]
        if data.get("dataset"): canonical["dataset"] = data["dataset"]
        if data.get("rank"): canonical["rank"] = str(data["rank"])
        if data.get("status"): canonical["status"] = str(data["status"])
        
        # Nested structures (e.g. full_metrics.json format)
        perf = data.get("performance", {})
        eff = data.get("efficiency", {})
        hw = data.get("hardware", {})
        
        if isinstance(perf, dict):
            if "accuracy_pct" in perf: canonical["metrics"]["accuracy"] = perf["accuracy_pct"]
            elif "accuracy" in perf: canonical["metrics"]["accuracy"] = perf["accuracy"]
            if "f1_score" in perf: canonical["metrics"]["f1"] = perf["f1_score"]
            elif "f1" in perf: canonical["metrics"]["f1"] = perf["f1"]
            if "val_loss" in perf: canonical["metrics"]["loss"] = perf["val_loss"]
            elif "loss" in perf: canonical["metrics"]["loss"] = perf["loss"]
            if "elapsed_s" in perf: canonical["resources"]["duration_s"] = perf["elapsed_s"]
            if "status" in perf and canonical["status"] == "PASS": canonical["status"] = perf["status"]
            
        if isinstance(eff, dict):
            if "compression_rate" in eff: canonical["resources"]["compression_ratio"] = eff["compression_rate"]
            elif "compression_ratio" in eff: canonical["resources"]["compression_ratio"] = eff["compression_ratio"]
            if "per_layer_ranks" in eff and not canonical["rank"]:
                ranks = list(eff["per_layer_ranks"].values())
                if ranks: canonical["rank"] = str(ranks[0])
                
        if isinstance(hw, dict):
            if "peak_memory_mb" in hw: canonical["resources"]["peak_vram_mb"] = hw["peak_memory_mb"]
            elif "peak_vram_mb" in hw: canonical["resources"]["peak_vram_mb"] = hw["peak_vram_mb"]

        # Top-level metric keys
        for key in ["accuracy", "f1", "loss", "perplexity", "eval_loss", "eval_accuracy"]:
            if key in data and key not in canonical["metrics"]:
                canonical["metrics"][key] = data[key]
        
        for key in ["peak_vram_mb", "latency_ms", "throughput_tok_per_sec", "duration_s", "compression_ratio"]:
            if key in data and key not in canonical["resources"]:
                canonical["resources"][key] = data[key]
                
    elif isinstance(data, list):
        # List payloads (e.g. summaries)
        canonical["status"] = "PASS"
        if data and isinstance(data[0], dict):
            first = data[0]
            if "accuracy" in first: canonical["metrics"]["accuracy"] = first["accuracy"]
            if "loss" in first: canonical["metrics"]["loss"] = first["loss"]
            if "status" in first: canonical["status"] = first["status"]

    return canonical

def main():
    all_files = sorted(Path("results/").rglob("*.json"))
    all_files = [f for f in all_files if "_archive" not in str(f) and not f.name.startswith("_") and f.name != "REORGANIZATION_MANIFEST.json"]
    
    normalized = []
    for f in all_files:
        result = normalize_file(f)
        if result:
            normalized.append(result)
            
    with open("results/_MASTER_SUMMARY_V2.csv", "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(CANONICAL_FIELDS)
        for r in normalized:
            writer.writerow([
                r.get("source_file", ""),
                r.get("module", ""),
                r.get("model", ""),
                r.get("dataset", ""),
                r.get("rank", ""),
                r.get("metrics", {}).get("accuracy", ""),
                r.get("metrics", {}).get("f1", ""),
                r.get("metrics", {}).get("loss", ""),
                r.get("metrics", {}).get("perplexity", ""),
                r.get("resources", {}).get("peak_vram_mb", ""),
                r.get("resources", {}).get("latency_ms", ""),
                r.get("resources", {}).get("duration_s", ""),
                r.get("resources", {}).get("compression_ratio", ""),
                r.get("status", ""),
            ])
            
    print(f"Normalized {len(normalized)} files")
    print("Wrote results/_MASTER_SUMMARY_V2.csv")

if __name__ == "__main__":
    main()
