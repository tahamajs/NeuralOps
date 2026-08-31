import csv
import json
import math
from collections import defaultdict

csv_path = r"D:\SVD_Project\results\final_paper_run\summary.csv"

def analyze():
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    sst2_rows = [r for r in all_rows if r["dataset"].strip().upper() == "SST2"]
    
    print(f"Total SST-2 rows: {len(sst2_rows)}")
    status_counts = defaultdict(int)
    for r in sst2_rows:
        status_counts[r["status"].strip().upper()] += 1
    
    print(f"Status breakdown: {dict(status_counts)}")
    
    pass_rows = [r for r in sst2_rows if r["status"].strip().upper() == "PASS"]
    fail_rows = [r for r in sst2_rows if r["status"].strip().upper() == "FAIL"]
    timeout_rows = [r for r in sst2_rows if r["status"].strip().upper() == "TIMEOUT"]
    
    def stats(vals):
        if not vals:
            return {}
        vals = sorted(vals)
        n = len(vals)
        mean_v = sum(vals) / n
        median_v = vals[n // 2] if n % 2 != 0 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
        var_v = sum((x - mean_v) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
        std_v = math.sqrt(var_v)
        return {
            "count": n,
            "mean": round(mean_v, 2),
            "median": round(median_v, 2),
            "std": round(std_v, 2),
            "min": round(vals[0], 2),
            "max": round(vals[-1], 2),
        }
    
    metrics = {
        "Accuracy": [float(r["mean_accuracy"]) for r in pass_rows],
        "F1": [float(r["f1_score"]) for r in pass_rows],
        "Fidelity": [float(r["fidelity_score"]) for r in pass_rows],
        "Throughput": [float(r["throughput_tokens_sec"]) for r in pass_rows],
        "VRAM": [float(r["peak_vram_mb"]) for r in pass_rows],
    }
    
    print("\n--- 2. DESCRIPTIVE STATISTICS (PASS Modules, N={}) ---".format(len(pass_rows)))
    for name, v in metrics.items():
        st = stats(v)
        print(f"{name:12s}: Mean={st['mean']}, Median={st['median']}, Std={st['std']}, Min={st['min']}, Max={st['max']}")
        
    ranges = [
        ("49-60%", 49.0, 60.0),
        ("60-70%", 60.0, 70.0),
        ("70-80%", 70.0, 80.0),
        ("80-90%", 80.0, 90.0),
        ("90-100%", 90.0, 100.01),
    ]
    range_counts = defaultdict(int)
    for a in metrics["Accuracy"]:
        for label, lo, hi in ranges:
            if lo <= a < hi:
                range_counts[label] += 1
                break
    print("\n--- ACCURACY HISTOGRAM BREAKDOWN ---")
    for label, _, _ in ranges:
        cnt = range_counts[label]
        print(f"  {label:10s}: {cnt:3d} modules ({cnt/len(pass_rows)*100:5.1f}%)")
        
    # 3. Classification
    tiers = {
        "Excellent (>=90%)": [],
        "Good (85-90%)": [],
        "Moderate (75-85%)": [],
        "Weak (60-75%)": [],
        "Collapsed (<60%)": [],
    }
    
    for r in pass_rows:
        acc = float(r["mean_accuracy"])
        f1 = float(r["f1_score"])
        fid = float(r["fidelity_score"])
        tp = float(r["throughput_tokens_sec"])
        vr = float(r["peak_vram_mb"])
        item = {
            "module": r["module_name"],
            "accuracy": acc,
            "f1": f1,
            "fidelity": fid,
            "throughput": tp,
            "vram": vr,
        }
        if acc >= 90.0:
            tiers["Excellent (>=90%)"].append(item)
        elif acc >= 85.0:
            tiers["Good (85-90%)"].append(item)
        elif acc >= 75.0:
            tiers["Moderate (75-85%)"].append(item)
        elif acc >= 60.0:
            tiers["Weak (60-75%)"].append(item)
        else:
            tiers["Collapsed (<60%)"].append(item)
            
    print("\n--- 3. MODULE CLASSIFICATION BY ACCURACY TIER ---")
    for tname, items in tiers.items():
        items.sort(key=lambda x: x["accuracy"], reverse=True)
        print(f"\n>> {tname} — Count: {len(items)}")
        for it in items:
            print(f"  {it['module']:30s} | Acc: {it['accuracy']:5.2f}% | F1: {it['f1']:.4f} | Fid: {it['fidelity']:.4f} | TP: {it['throughput']:7.1f} tok/s | VRAM: {it['vram']:6.1f} MB")
            
    # 4. Baseline Comparison
    full_row = next((r for r in pass_rows if r["module_name"].strip() == "full"), None)
    if full_row:
        base_acc = float(full_row["mean_accuracy"])
        base_fid = float(full_row["fidelity_score"])
        print(f"\n--- 4. BASELINE COMPARISON (full baseline Acc={base_acc}%, Fid={base_fid}) ---")
        
        near_baseline = []
        for r in pass_rows:
            acc = float(r["mean_accuracy"])
            fid = float(r["fidelity_score"])
            diff_acc = round(acc - base_acc, 2)
            diff_fid = round(fid - base_fid, 4)
            if fid > 0.90 and abs(diff_acc) <= 2.0:
                near_baseline.append((r["module_name"], acc, diff_acc, fid, diff_fid, float(r["throughput_tokens_sec"])))
        
        print(f"Modules with Fidelity > 0.90 AND Accuracy within 2.0% of Baseline ({len(near_baseline)} modules):")
        near_baseline.sort(key=lambda x: x[1], reverse=True)
        for name, acc, d_acc, fid, d_fid, tp in near_baseline:
            print(f"  {name:30s} | Acc: {acc:5.2f}% (diff: {d_acc:+5.2f}%) | Fid: {fid:.4f} (diff: {d_fid:+.4f}) | TP: {tp:7.1f} tok/s")
            
    # 5. Families Classification
    def classify_family(name):
        n = name.lower()
        if any(k in n for k in ["svd", "spectral", "rpca", "pca", "asvd", "saes", "subspace", "grecodec"]):
            return "SVD & Spectral Factorization"
        elif any(k in n for k in ["moe", "expert", "tomoe", "bam", "reap"]):
            return "Mixture-of-Experts (MoE)"
        elif any(k in n for k in ["ssm", "mamba", "hgrn", "rnn", "griffin", "s4", "samba", "simple_ssm"]):
            return "Linear RNN & State Space Models (SSM)"
        elif any(k in n for k in ["prun", "proxsparse", "sparse", "flap", "slice", "slim", "bonsai", "lsc"]):
            return "Pruning & Sparsity"
        elif any(k in n for k in ["quant", "kmeans", "int4", "fp8", "coopq", "hash", "admm"]):
            return "Quantization & Hash Compression"
        elif any(k in n for k in ["cur"]):
            return "CUR Decompositions"
        elif any(k in n for k in ["rope", "gqa", "wgqa", "kv", "cachenotes", "zerokv", "hybridkv", "tpa"]):
            return "KV Cache & Architectural Attention"
        elif any(k in n for k in ["flash", "cosformer", "fasa", "faq", "eat", "cluster", "astro", "nystrom", "performer", "linear", "linformer", "galerkin", "infini", "mixattention", "moba", "nsa", "nexus", "leka", "aegis", "enlighten", "span", "block", "fusion", "compot", "come", "dobi", "kronecker", "tt", "lsh", "modegpt", "nli", "nmf", "oats", "oha", "orion", "phantom", "rala", "regla", "rpc", "salr", "salt", "sdrm", "soka", "spectron", "squeezing", "uniql", "vig", "vspec", "xattention", "zeros", "liftr", "lillama", "lola", "mla", "primal", "hymba", "jamba"]):
            return "Efficient Attention & Kernelization"
        else:
            return "Other / Hybrid"

    family_map = defaultdict(list)
    for r in pass_rows:
        fam = classify_family(r["module_name"])
        family_map[fam].append(r)
        
    print("\n--- 5. ARCHITECTURAL FAMILIES ON SST-2 ---")
    family_summary = []
    for fam, frows in family_map.items():
        fam_accs = [float(r["mean_accuracy"]) for r in frows]
        fam_fids = [float(r["fidelity_score"]) for r in frows]
        fam_tps = [float(r["throughput_tokens_sec"]) for r in frows]
        fam_vrams = [float(r["peak_vram_mb"]) for r in frows]
        
        avg_acc = sum(fam_accs) / len(fam_accs)
        avg_fid = sum(fam_fids) / len(fam_fids)
        avg_tp = sum(fam_tps) / len(fam_tps)
        avg_vr = sum(fam_vrams) / len(fam_vrams)
        family_summary.append({
            "family": fam,
            "count": len(frows),
            "avg_accuracy": round(avg_acc, 2),
            "avg_fidelity": round(avg_fid, 4),
            "avg_throughput": round(avg_tp, 1),
            "avg_vram": round(avg_vr, 1),
        })
        
    family_summary.sort(key=lambda x: x["avg_accuracy"], reverse=True)
    for fs in family_summary:
        print(f"  {fs['family']:42s} | Count: {fs['count']:2d} | Avg Acc: {fs['avg_accuracy']:5.2f}% | Avg Fid: {fs['avg_fidelity']:.4f} | Avg TP: {fs['avg_throughput']:7.1f} tok/s | Avg VRAM: {fs['avg_vram']:6.1f} MB")
        
    # 6. Failure Analysis
    print(f"\n--- 6. FAILURES & TIMEOUTS ON SST-2 (Total FAIL={len(fail_rows)}, TIMEOUT={len(timeout_rows)}) ---")
    for r in fail_rows:
        print(f"  FAIL: {r['module_name']}")
    for r in timeout_rows:
        print(f"  TIMEOUT: {r['module_name']}")
        
    # Save structured summary as JSON
    out_data = {
        "total_evaluated": len(sst2_rows),
        "status_counts": dict(status_counts),
        "descriptive_stats": {name: stats(v) for name, v in metrics.items()},
        "range_counts": dict(range_counts),
        "tiers": tiers,
        "family_summary": family_summary,
        "failures": [r["module_name"] for r in fail_rows],
        "timeouts": [r["module_name"] for r in timeout_rows],
    }
    with open(r"D:\SVD_Project\results\final_paper_run\sst2_analysis_summary.json", "w") as f:
        json.dump(out_data, f, indent=2)
    print("\nSaved D:\\SVD_Project\\results\\final_paper_run\\sst2_analysis_summary.json")

if __name__ == "__main__":
    analyze()
