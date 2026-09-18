import os
import json
import glob

files = sorted(glob.glob("results/all_modules/*/*full_metrics.json"))
aegis_files = sorted(glob.glob("results/fixed_aegis/*/*full_metrics.json"))

all_files = aegis_files + files

print(f"{'Module':<30} | {'Params Before':<14} | {'Params After':<14} | {'Delta':<14} | {'Comp. Layers':<12} | {'Category':<14} | {'Real Comp.?':<11}")
print("-" * 115)

compressors = []
inflated = []
adapter_only = []
broken = []

for f in all_files:
    try:
        with open(f, 'r') as fp:
            d = json.load(fp)
    except Exception:
        continue
    
    dirname = os.path.basename(os.path.dirname(f))
    mod = dirname.replace("_sst2", "").replace("_mnli", "").replace("_ag_news", "")
    
    eff = d.get("efficiency", {})
    perf = d.get("performance", {})
    
    p_before = eff.get("total_params_before", 0)
    p_after = eff.get("total_params_after", 0)
    delta = p_after - p_before
    comp_layers = eff.get("num_compressed_layers", 0)
    acc = perf.get("accuracy_pct", 0)
    
    pb_str = f"{p_before:,}"
    pa_str = f"{p_after:,}"
    delta_str = f"+{delta:,}" if delta > 0 else f"{delta:,}"
    
    # Classification rules:
    # BROKEN: delta > 50M or acc < 40% (except 4-class zero-shot defaults)
    # COMPRESSOR: delta < 0 and comp_layers > 0
    # INFLATED: delta > 0 and comp_layers > 0
    # ADAPTER_ONLY: comp_layers == 0
    if delta > 50000000 or (acc < 40.0 and "aegis_v3" not in mod and "aegis_v4" not in mod):
        cat = "BROKEN"
        broken.append(mod)
    elif comp_layers == 0:
        cat = "ADAPTER_ONLY"
        adapter_only.append(mod)
    elif delta < 0 and comp_layers > 0:
        cat = "COMPRESSOR"
        compressors.append(mod)
    else:
        cat = "INFLATED"
        inflated.append(mod)
        
    real_comp = "YES" if (delta < 0 and comp_layers > 0) else "NO"
    
    print(f"{mod:<30} | {pb_str:<14} | {pa_str:<14} | {delta_str:<14} | {comp_layers:<12} | {cat:<14} | {real_comp:<11}")

print("\n=== SUMMARY COUNTS ===")
print(f"COMPRESSOR:   {len(compressors)} -> {compressors}")
print(f"INFLATED:     {len(inflated)} -> {inflated}")
print(f"ADAPTER_ONLY: {len(adapter_only)} -> {adapter_only}")
print(f"BROKEN:       {len(broken)} -> {broken}")
