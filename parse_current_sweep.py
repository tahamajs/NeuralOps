import os
import json
import glob

print(f"{'Module':<28} | {'Model':<12} | {'Dataset':<8} | {'Status':<15} | {'Acc (%)':<8} | {'Comp.':<5} | {'Params Delta':<14} | {'VRAM (MB)':<10}")
print("-" * 115)

# 1. Fixed Aegis targets
aegis_files = glob.glob("results/fixed_aegis/*/*full_metrics.json")
for f in sorted(aegis_files):
    with open(f, 'r') as fp:
        data = json.load(fp)
    dirname = os.path.basename(os.path.dirname(f))
    
    if "mnli" in dirname: dataset = "mnli"; model = "roberta"
    elif "sst2" in dirname: dataset = "sst2"; model = "distilbert"
    elif "ag_news" in dirname: dataset = "ag_news"; model = "gpt2"
    else: dataset = "unknown"; model = "unknown"
    
    mod = dirname.replace("_" + dataset, "")
    perf = data.get("performance", {})
    eff = data.get("efficiency", {})
    hw = data.get("hardware", {})
    
    acc = perf.get("accuracy_pct", 0)
    status = perf.get("status", "UNKNOWN")
    comp_layers = eff.get("num_compressed_layers", 0)
    if comp_layers == 0 and eff.get("num_injected_attention_layers", 0) > 0:
        status = "ADAPTER_ONLY"
    
    p_before = eff.get("total_params_before", 0)
    p_after = eff.get("total_params_after", 0)
    delta = p_after - p_before
    delta_str = f"+{delta:,}" if delta > 0 else f"{delta:,}"
    vram = hw.get("peak_vram_mb", 0)
    
    print(f"{mod:<28} | {model:<12} | {dataset:<8} | {status:<15} | {acc:<8.2f} | {comp_layers:<5} | {delta_str:<14} | {vram:<10.1f}")

# 2. All newly run modules
all_mod_files = glob.glob("results/all_modules/*/*full_metrics.json")
for f in sorted(all_mod_files):
    with open(f, 'r') as fp:
        data = json.load(fp)
    dirname = os.path.basename(os.path.dirname(f))
    mod = dirname
    model = "distilbert"
    dataset = "sst2"
    
    perf = data.get("performance", {})
    eff = data.get("efficiency", {})
    hw = data.get("hardware", {})
    
    acc = perf.get("accuracy_pct", 0)
    status = perf.get("status", "UNKNOWN")
    comp_layers = eff.get("num_compressed_layers", 0)
    if comp_layers == 0 and eff.get("num_injected_attention_layers", 0) > 0:
        status = "ADAPTER_ONLY"
    
    p_before = eff.get("total_params_before", 0)
    p_after = eff.get("total_params_after", 0)
    delta = p_after - p_before
    delta_str = f"+{delta:,}" if delta > 0 else f"{delta:,}"
    vram = hw.get("peak_vram_mb", 0)
    
    print(f"{mod:<28} | {model:<12} | {dataset:<8} | {status:<15} | {acc:<8.2f} | {comp_layers:<5} | {delta_str:<14} | {vram:<10.1f}")
