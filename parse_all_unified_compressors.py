import os
import json
import glob

patterns = [
    "results/unified_*/distilbert/*full_metrics.json",
    "results/unified_*/distilbert_sst2*/*full_metrics.json",
    "results/unified_*/*full_metrics.json",
    "results/all_modules/*/*full_metrics.json",
    "results/fixed_aegis/*/*full_metrics.json"
]

all_files = []
for p in patterns:
    all_files.extend(glob.glob(p))

all_files = sorted(list(set(all_files)))

real_compressors = []

for f in all_files:
    if "full_full_metrics.json" in f or "baseline" in f:
        continue
    try:
        with open(f, 'r') as fp:
            d = json.load(fp)
    except Exception:
        continue

    eff = d.get("efficiency", {})
    perf = d.get("performance", {})
    
    p_before = eff.get("total_params_before", 0)
    p_after = eff.get("total_params_after", 0)
    delta = p_after - p_before
    comp_layers = eff.get("num_compressed_layers", 0)
    acc = perf.get("accuracy_pct", 0)
    
    mod_name = d.get("model", os.path.basename(f).replace("_full_metrics.json", ""))
    
    if delta < 0 and comp_layers > 0:
        ratio = p_before / p_after if p_after > 0 else 1.0
        real_compressors.append({
            "name": mod_name,
            "file": f,
            "acc": acc,
            "before": p_before,
            "after": p_after,
            "delta": delta,
            "ratio": ratio,
            "layers": comp_layers
        })

print(f"Total files checked: {len(all_files)}")
print(f"REAL COMPRESSORS FOUND (delta < 0 and comp_layers > 0): {len(real_compressors)}\n")

print(f"{'Module / Model':<40} | {'Acc (%)':<8} | {'Params Before':<14} | {'Params After':<14} | {'Delta':<14} | {'Ratio':<8} | {'Layers':<6}")
print("-" * 115)

real_compressors = sorted(real_compressors, key=lambda x: x["delta"])
for c in real_compressors:
    delta_str = f"{c['delta']:,}"
    pb_str = f"{c['before']:,}"
    pa_str = f"{c['after']:,}"
    print(f"{c['name']:<40} | {c['acc']:<8.2f} | {pb_str:<14} | {pa_str:<14} | {delta_str:<14} | {c['ratio']:<8.2f}x | {c['layers']:<6}")
