import os
import json
import glob

files = glob.glob("results/fixed_aegis/*/*full_metrics.json")
print(f"{'Module':<15} | {'Model':<15} | {'Dataset':<8} | {'Status':<15} | {'Acc':<6} | {'Comp.':<5} | {'Params Delta':<12}")
print("-" * 90)

for f in sorted(files):
    with open(f, 'r') as fp:
        data = json.load(fp)
    
    dirname = os.path.basename(os.path.dirname(f))
    parts = dirname.split("_")
    
    if "mnli" in dirname: dataset = "mnli"; model="roberta"
    elif "sst2" in dirname: dataset = "sst2"; model="distilbert"
    elif "ag_news" in dirname: dataset = "ag_news"; model="gpt2"
    else: dataset = "unknown"; model="unknown"
    
    mod = dirname.replace("_" + dataset, "")
        
    perf = data.get("performance", {})
    eff = data.get("efficiency", {})
    
    acc = perf.get("accuracy_pct", 0)
    status = perf.get("status", "UNKNOWN")
    comp_layers = eff.get("num_compressed_layers", 0)
    if comp_layers == 0 and eff.get("num_injected_attention_layers", 0) > 0:
        status = "ADAPTER_ONLY"
    
    params_base = eff.get("total_params_before", 0)
    params_comp = eff.get("total_params_after", 0)
    delta = params_comp - params_base
    
    delta_str = f"+{delta}" if delta > 0 else str(delta)
    if delta == 0 and params_base == 0:
        delta_str = "N/A"
        
    print(f"{mod:<15} | {model:<15} | {dataset:<8} | {status:<15} | {acc:<6.2f} | {comp_layers:<5} | {delta_str:<12}")
