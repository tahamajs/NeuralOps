import os
import sys
import json
import csv
from collections import defaultdict, Counter

sys.path.insert(0, r"D:\SVD_Project\Codes")

def main():
    print("="*70)
    print("COMPREHENSIVE BENCHMARK PROJECT AUDIT (GPU SERVER)")
    print("="*70)
    
    # 1. Config & Module List
    config_path = r"D:\SVD_Project\results\final_paper_run\config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)
    modules = cfg["modules"]
    datasets = cfg["datasets"]
    total_expected_runs = len(modules) * len(datasets)
    print(f"\n[1] Config Info:")
    print(f"  Total configured modules: {len(modules)}")
    print(f"  Configured datasets: {datasets}")
    print(f"  Expected total runs: {total_expected_runs}")
    
    # 2. Registry Coverage
    try:
        from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
        import framework.attentions
        import framework.compression_engine
        
        attn_keys = set(ATTENTION_REGISTRY.keys())
        comp_keys = set(COMPRESSOR_REGISTRY.keys())
        all_keys = attn_keys.union(comp_keys)
        
        missing_registry = []
        for m in modules:
            m_lower = m.lower()
            found = (m_lower in all_keys or f"{m_lower}_inject" in all_keys or f"{m_lower}_compressor" in all_keys or m_lower.replace("-", "_") in all_keys or m_lower in ATTENTION_REGISTRY or m_lower in COMPRESSOR_REGISTRY)
            if not found:
                missing_registry.append(m)
        
        print(f"\n[2] Registry Coverage:")
        print(f"  Attention registry entries: {len(attn_keys)}")
        print(f"  Compressor registry entries: {len(comp_keys)}")
        print(f"  Unregistered modules from config: {missing_registry} (Count: {len(missing_registry)})")
    except Exception as e:
        print(f"\n[2] Registry Import Error: {e}")
    
    # 3. Results CSV Data Integrity
    csv_path = r"D:\SVD_Project\results\final_paper_run\summary.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\n[3] Results Data Integrity:")
    print(f"  Total completed rows in summary.csv: {len(rows)}")
    
    dataset_counts = Counter(r["dataset"] for r in rows)
    print(f"  Breakdown by dataset: {dict(dataset_counts)}")
    
    status_counts = Counter((r["dataset"], r["status"]) for r in rows)
    print(f"  Breakdown by (dataset, status):")
    for k, v in sorted(status_counts.items()):
        print(f"    {k[0]} - {k[1]}: {v}")
    
    # Check duplicate runs
    seen = Counter((r["module_name"], r["dataset"]) for r in rows)
    duplicates = [k for k, v in seen.items() if v > 1]
    print(f"  Duplicate (module, dataset) entries: {duplicates} (Count: {len(duplicates)})")
    
    # Check out-of-range metrics
    anomalies = []
    for r in rows:
        m = r["module_name"]
        d = r["dataset"]
        st = r["status"]
        try:
            acc = float(r["mean_accuracy"])
            f1 = float(r.get("f1_score", 0.0))
            vram = float(r.get("peak_memory_mb", 0.0))
            if st == "PASS":
                if not (0.0 <= acc <= 100.0):
                    anomalies.append((m, d, "acc out of range", acc))
                if not (0.0 <= f1 <= 1.0):
                    anomalies.append((m, d, "f1 out of range", f1))
        except ValueError as e:
            anomalies.append((m, d, "parse error", str(e)))
    print(f"  Metric anomalies count: {len(anomalies)}")
    
    # 4. Low-Accuracy Modules Analysis (MRPC)
    mrpc_rows = [r for r in rows if r["dataset"] == "MRPC" and r["status"] == "PASS"]
    low_acc_mrpc = [r for r in mrpc_rows if float(r["mean_accuracy"]) < 55.0]
    print(f"\n[4] Low-Accuracy Modules on MRPC (<55%): {len(low_acc_mrpc)}")
    by_acc = defaultdict(list)
    for r in low_acc_mrpc:
        by_acc[f"{float(r['mean_accuracy']):.2f}%"].append(r["module_name"])
    for acc_val, mods in sorted(by_acc.items()):
        print(f"    Accuracy {acc_val} ({len(mods)} modules): {', '.join(mods)}")
    
    # 5. Failed & Timeout Modules List
    fails_mrpc = [r["module_name"] for r in rows if r["dataset"] == "MRPC" and r["status"] == "FAIL"]
    timeouts_mrpc = [r["module_name"] for r in rows if r["dataset"] == "MRPC" and r["status"] == "TIMEOUT"]
    fails_sst2 = [r["module_name"] for r in rows if r["dataset"] == "SST2" and r["status"] == "FAIL"]
    timeouts_sst2 = [r["module_name"] for r in rows if r["dataset"] == "SST2" and r["status"] == "TIMEOUT"]
    
    print(f"\n[5] Failures & Timeouts Summary:")
    print(f"  SST2 FAIL ({len(fails_sst2)}): {fails_sst2}")
    print(f"  SST2 TIMEOUT ({len(timeouts_sst2)}): {timeouts_sst2}")
    print(f"  MRPC FAIL ({len(fails_mrpc)}): {fails_mrpc}")
    print(f"  MRPC TIMEOUT ({len(timeouts_mrpc)}): {timeouts_mrpc}")
    
    # 6. Execution Progress
    evaluated_mrpc = {r["module_name"] for r in rows if r["dataset"] == "MRPC"}
    remaining_mrpc = [m for m in modules if m not in evaluated_mrpc]
    print(f"\n[6] Execution Progress:")
    print(f"  SST2: {dataset_counts.get('SST2', 0)} / {len(modules)} ({dataset_counts.get('SST2', 0)/len(modules)*100:.1f}%)")
    print(f"  MRPC: {dataset_counts.get('MRPC', 0)} / {len(modules)} ({dataset_counts.get('MRPC', 0)/len(modules)*100:.1f}%)")
    print(f"  Overall: {len(rows)} / {total_expected_runs} ({len(rows)/total_expected_runs*100:.1f}%)")
    print(f"  Remaining MRPC modules ({len(remaining_mrpc)}): {remaining_mrpc[:10]} ... (+ {max(0, len(remaining_mrpc)-10)} more)")
    
    # 7. Code Fixes Verification on Disk
    print(f"\n[7] Code Fixes Verification on Disk:")
    
    checks = [
        (r"D:\SVD_Project\Codes\framework\attentions\decay_linear_attention.py", "torch.pow", "Vectorized decay matrix (decay_linear_attn)"),
        (r"D:\SVD_Project\Codes\framework\attentions\simple_ssm.py", "torch.pow(dA_mean", "SSM vectorized transition kernel (simple_ssm)"),
        (r"D:\SVD_Project\Codes\framework\attentions\hgrn2_layer.py", "return out, None", "2-tuple return signature (hgrn2_layer)"),
        (r"D:\SVD_Project\Codes\framework\attentions\hgrn2_gated_deltanet.py", "return out, None", "2-tuple return signature (hgrn2_gated_deltanet)"),
        (r"D:\SVD_Project\Codes\framework\attentions\hgrn2_outer.py", "dim = 768", "768 dim default (hgrn2_outer)"),
        (r"D:\SVD_Project\Codes\framework\attentions\hla_attention.py", "num_heads = 12", "12 heads default (hla_attention)"),
        (r"D:\SVD_Project\Codes\framework\attentions\vig_attention.py", "isinstance(config, int)", "ViG config normalization (vig)"),
        (r"D:\SVD_Project\Codes\framework\attentions\cosformer.py", "F.elu", "Smooth ELU+1 kernel (cosformer)"),
        (r"D:\SVD_Project\Codes\framework\attentions\aegis_v1.py", "torch.linalg.qr", "ORF orthogonal random features (aegis_v1)"),
        (r"D:\SVD_Project\Codes\framework\attentions\mamba.py", "copy_weights", "Pretrained weights copy + pres_gate (mamba)"),
        (r"D:\SVD_Project\Codes\framework\attentions\leka_v1.py", "persistent=True", "Persistent QR buffer + pres_gate (leka_v1)"),
        (r"D:\SVD_Project\Codes\framework\attentions\minimal_aurora.py", "UniversalAdapter", "DistilBERT direct fallback (minimal_aurora)"),
        (r"D:\SVD_Project\Codes\framework\attentions\minimal_nova.py", "UniversalAdapter", "DistilBERT direct fallback (minimal_nova)"),
        (r"D:\SVD_Project\Codes\framework\attentions\nexus.py", "torch.ones(B, H, n_blocks, n_blocks", "Full NxN Seer mask + QR RFF (nexus)"),
        (r"D:\SVD_Project\Codes\framework\compression_engine\cur_moe_compressor.py", "_ATTN_NAMES", "Skip attention projections in CUR (cur_moe)"),
        (r"D:\SVD_Project\Codes\framework\compression_engine\lillama_compressor.py", "95% energy guard", "95% energy threshold + skip attn (lillama)"),
    ]
    
    all_fixes_passed = True
    for path, target_str, desc in checks:
        if not os.path.exists(path):
            print(f"  [FAIL] {desc}: File not found at {path}")
            all_fixes_passed = False
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if target_str in content:
            print(f"  [PASS] {desc}")
        else:
            print(f"  [FAIL] {desc}: Target substring '{target_str}' missing")
            all_fixes_passed = False
    
    print(f"\n  Overall Code Fix Status: {'ALL CODE FIXES VERIFIED & SYNCED (100% PASS)' if all_fixes_passed else 'SOME FIXES MISSING'}")
    print("="*70)

if __name__ == "__main__":
    main()
