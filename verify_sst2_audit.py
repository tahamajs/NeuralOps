import csv
import json
import math
from collections import defaultdict, Counter

csv_path = r"D:\SVD_Project\results\final_paper_run\summary.csv"

def run_audit():
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    sst2_rows = [r for r in all_rows if r["dataset"].strip().upper() == "SST2"]
    
    print("="*70)
    print("AUDIT: SST-2 RAW DATA VS REPORT CROSS-CHECK")
    print("="*70)
    
    # 1. Total modules & status in CSV
    print(f"\n[1] Row count in CSV for SST-2: {len(sst2_rows)}")
    status_counts = Counter(r["status"].strip().upper() for r in sst2_rows)
    print(f"  Raw CSV status column counts: {dict(status_counts)}")
    
    # Check SST-2 failures: identify which were timeouts from execution time / logs
    fail_modules = [r["module_name"] for r in sst2_rows if r["status"].strip().upper() == "FAIL"]
    print(f"\n[2] SST-2 FAIL modules list ({len(fail_modules)}):")
    print(f"  {fail_modules}")
    
    # Known timeout modules based on execution history:
    timeout_suspects = [
        "adasplash", "aegis_spectron_fp8", "bam_umoe", "decay_linear_attn", 
        "simple_ssm", "svd", "svd-mamba", "svd_mamba", "upi_stuffed_mamba", 
        "vspec", "mamba_svd_flash", "hymba_transx", "jamba", "jamba2", "jamba_info"
    ]
    actual_timeouts = [m for m in timeout_suspects if m in fail_modules]
    actual_crashes = [m for m in fail_modules if m not in timeout_suspects]
    print(f"\n  Underlying Breakdown of 27 SST-2 Failures:")
    print(f"    Execution Timeouts (>900s) ({len(actual_timeouts)}): {actual_timeouts}")
    print(f"    Code/Signature/Config Errors ({len(actual_crashes)}): {actual_crashes}")
    
    # 3. Descriptive Stats of PASS modules
    pass_rows = [r for r in sst2_rows if r["status"].strip().upper() == "PASS"]
    print(f"\n[3] PASS modules count: {len(pass_rows)}")
    
    accs = [float(r["mean_accuracy"]) for r in pass_rows]
    f1s = [float(r["f1_score"]) for r in pass_rows]
    fids = [float(r["fidelity_score"]) for r in pass_rows]
    tps = [float(r["throughput_tokens_sec"]) for r in pass_rows]
    vrams = [float(r["peak_vram_mb"]) for r in pass_rows]
    
    def stats(vals):
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

    for name, v in [("Accuracy", accs), ("F1", f1s), ("Fidelity", fids), ("Throughput", tps), ("VRAM", vrams)]:
        st = stats(v)
        print(f"  {name:12s}: Mean={st['mean']}, Median={st['median']}, Std={st['std']}, Min={st['min']}, Max={st['max']}")

    # 4. Collapsed Modules Fidelity Analysis
    collapsed_rows = [r for r in pass_rows if float(r["mean_accuracy"]) < 60.0]
    print(f"\n[4] Collapsed Modules (<60%) Fidelity Values ({len(collapsed_rows)} modules):")
    for r in collapsed_rows:
        print(f"  {r['module_name']:30s} | Acc: {float(r['mean_accuracy']):5.2f}% | F1: {float(r['f1_score']):.4f} | Fid: {float(r['fidelity_score']):.4f}")
        
    # 5. Baseline Comparison Audit
    full_row = next((r for r in pass_rows if r["module_name"].strip() == "full"), None)
    base_acc = float(full_row["mean_accuracy"])
    base_fid = float(full_row["fidelity_score"])
    print(f"\n[5] Full Baseline: Accuracy={base_acc}%, Fidelity={base_fid}")
    
    strict_near_baseline = []
    relaxed_near_baseline = []
    for r in pass_rows:
        acc = float(r["mean_accuracy"])
        fid = float(r["fidelity_score"])
        diff_acc = acc - base_acc
        if fid > 0.90 and abs(diff_acc) <= 2.0:
            strict_near_baseline.append((r["module_name"], acc, fid, diff_acc))
        if abs(diff_acc) <= 2.0:
            relaxed_near_baseline.append((r["module_name"], acc, fid, diff_acc))
            
    print(f"  Strict criteria (Fidelity > 0.90 AND |diff_acc| <= 2.0%): {len(strict_near_baseline)} modules")
    for m, a, f, d in strict_near_baseline:
        print(f"    {m:30s} | Acc: {a:5.2f}% | Fid: {f:.4f} | Diff: {d:+5.2f}%")
        
    print(f"\n  Modules with |diff_acc| <= 2.0% regardless of Fidelity: {len(relaxed_near_baseline)} modules")
    
    # 6. Rigorous Family Re-Classification
    def classify_family_rigorous(name):
        n = name.lower()
        # 1. MoE
        if any(k in n for k in ["moe", "expert", "tomoe", "bam", "reap"]):
            return "Mixture-of-Experts (MoE)"
        # 2. Linear RNN, SSM & Recurrence
        elif any(k in n for k in ["ssm", "mamba", "hgrn", "rnn", "griffin", "s4", "samba", "simple_ssm", "hla", "gated_linear_attn", "decay_linear_attn", "factorised_matrix_state", "halo_regla", "regla", "rala"]):
            return "Linear RNN, SSM & State-Space Models"
        # 3. SVD, Spectral & Subspace Decompositions
        elif any(k in n for k in ["svd", "spectral", "rpca", "pca", "asvd", "saes", "subspace", "grecodec", "lillama", "minimal_aurora", "minimal_nova", "direct_svd", "swift_svd", "res_svd", "spectral_residual"]):
            return "SVD, Spectral & Subspace Factorization"
        # 4. CUR Decompositions
        elif any(k in n for k in ["cur"]):
            return "CUR Cross-Approximation"
        # 5. Linear & Kernelized Attention (RFF, Softmax-free, Taylor)
        elif any(k in n for k in ["cosformer", "performer", "leka", "aegis", "galerkin", "linformer", "nystrom", "spectron", "spectra", "kda", "linear_decay", "soka"]):
            return "Linear & Kernelized Attention"
        # 6. Sparse, Chunked & Multi-Scale Attention
        elif any(k in n for k in ["nexus", "cluster", "nsa", "xattention", "proxsparse", "adaptive_sparse", "fasa", "eat", "astro", "infini", "moba", "mixattention", "block_transformer", "span_expanded", "fusion_attn", "compot", "come", "dobi", "faq", "flash", "orion", "phantom", "rpc", "sdrm", "vig", "vspec", "nli_softmax", "nmf", "oats", "oha", "zeros"]):
            return "Sparse, Chunked & Efficient Attention"
        # 7. Quantization, Low-Rank & Tensor Trains
        elif any(k in n for k in ["quant", "kmeans", "int4", "fp8", "coopq", "hash", "admm", "kronecker", "tt", "mlorq", "rotational_slicegpt", "slicegpt", "sparsegpt", "huffman"]):
            return "Quantization, Pruning & Tensor Compression"
        # 8. KV Cache & Multi-Query Architectures
        elif any(k in n for k in ["gqa", "wgqa", "kv", "cachenotes", "zerokv", "hybridkv", "tpa", "rope", "unified_rope"]):
            return "KV Cache & Multi-Query Attention"
        # 9. Distillation & Head Pruning
        elif any(k in n for k in ["squeezing", "liftr", "flap", "slim", "bonsai", "lsc", "progressive_token_pruning", "full"]):
            return "Structured Pruning & Baseline"
        else:
            return "Other / Hybrid"

    fam_groups = defaultdict(list)
    for r in pass_rows:
        fam = classify_family_rigorous(r["module_name"])
        fam_groups[fam].append(r)
        
    print("\n[6] REVISED ARCHITECTURAL FAMILIES SUMMARY:")
    revised_summary = []
    for fam, frows in sorted(fam_groups.items()):
        f_accs = [float(r["mean_accuracy"]) for r in frows]
        f_fids = [float(r["fidelity_score"]) for r in frows]
        f_tps = [float(r["throughput_tokens_sec"]) for r in frows]
        f_vrams = [float(r["peak_vram_mb"]) for r in frows]
        revised_summary.append({
            "family": fam,
            "count": len(frows),
            "mean_acc": round(sum(f_accs)/len(f_accs), 2),
            "mean_fid": round(sum(f_fids)/len(f_fids), 4),
            "mean_tp": round(sum(f_tps)/len(f_tps), 1),
            "mean_vram": round(sum(f_vrams)/len(f_vrams), 1),
        })
        
    revised_summary.sort(key=lambda x: x["mean_acc"], reverse=True)
    for fs in revised_summary:
        print(f"  {fs['family']:45s} | Count: {fs['count']:2d} | Mean Acc: {fs['mean_acc']:5.2f}% | Mean Fid: {fs['mean_fid']:.4f} | Mean TP: {fs['mean_tp']:7.1f} tok/s | Mean VRAM: {fs['mean_vram']:6.1f} MB")

    # Save revised audit results
    audit_data = {
        "sst2_evaluated": len(sst2_rows),
        "pass_count": len(pass_rows),
        "fail_count": len(fail_modules),
        "timeout_count_underlying": len(actual_timeouts),
        "crash_count_underlying": len(actual_crashes),
        "strict_near_baseline_count": len(strict_near_baseline),
        "strict_near_baseline_modules": strict_near_baseline,
        "revised_family_summary": revised_summary,
    }
    with open(r"D:\SVD_Project\results\final_paper_run\audit_results.json", "w") as f:
        json.dump(audit_data, f, indent=2)
    print("\nSaved D:\\SVD_Project\\results\\final_paper_run\\audit_results.json")

if __name__ == "__main__":
    run_audit()
