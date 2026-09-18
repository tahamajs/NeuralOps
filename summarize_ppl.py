import json
import csv
from pathlib import Path

categories = {
    "lora_attention": "Category 1", "lsh_attention": "Category 1", "compressed_mha": "Category 1", "gqa": "Category 1", "aegis_hybrid": "Category 1",
    "full_attention": "Category 2", "svd_linear_attention": "Category 2", "reap_moe": "Category 2", "minimal_prism": "Category 2", "halo_regla": "Category 2",
    "cosformer": "Category 3", "performer_attention": "Category 3", "samba_m3": "Category 3", "mamba": "Category 3", "hla_inject": "Category 3",
    "gated_deltanet_block": "Category 3", "galerkin": "Category 3", "phantom": "Category 3", "jamba": "Category 3", "unified_rope": "Category 3",
    "adaptive_svd_attention": "Category 4", "moba": "Category 4", "mla_attention": "Category 4", "vig_attention": "Category 4", "zeros_attention": "Category 4",
    "aasvd": "Category 5", "slicegpt": "Category 5", "sparsegpt": "Category 5", "proxsparse_pruner": "Category 5", "saes_svd": "Category 5"
}

baseline_file = Path("results/wikitext2_baseline/gpt2/ppl_result.json")
baseline_ppl = 27.3236
if baseline_file.exists():
    try:
        baseline_ppl = json.loads(baseline_file.read_text()).get("perplexity", 27.3236)
    except Exception:
        pass

rows = []
for mod, cat in categories.items():
    res_file = Path(f"results/wikitext2_{mod}/gpt2/ppl_result.json")
    ppl = None
    if res_file.exists():
        try:
            data = json.loads(res_file.read_text())
            ppl = data.get("perplexity")
        except Exception:
            pass
    
    delta = (ppl - baseline_ppl) if ppl is not None else None
    rows.append({
        "module": mod,
        "perplexity": round(ppl, 4) if ppl is not None else "FAIL/OOM",
        "delta_vs_baseline": round(delta, 4) if delta is not None else "N/A",
        "category": cat
    })

out_csv = Path("audit/wikitext2_ppl_summary.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["module", "perplexity", "delta_vs_baseline", "category"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {out_csv} (baseline PPL: {baseline_ppl:.4f})")
