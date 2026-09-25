import glob
import csv
import os
from pathlib import Path

BASE = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(".")
results = []

for p in sorted(glob.glob(str(BASE / "results" / "rank_sweep_*" / "*" / "*" / "summary.csv"))):
    parts = Path(p).parts
    # parts: (..., 'results', 'rank_sweep_sst2', 'a3_compressor', 'rank_16', 'summary.csv')
    try:
        model_bench = parts[-4]
        module = parts[-3]
        rank_str = parts[-2]
    except Exception:
        continue

    try:
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row.get("status", "")
                acc = row.get("accuracy_pct", "")
                ppl = row.get("perplexity", "")
                f1 = row.get("f1_score", "")
                comp = row.get("compression_rate", "")
                recon = row.get("avg_reconstruction_error", "")
                results.append({
                    "bench": model_bench,
                    "module": module,
                    "rank": rank_str,
                    "status": status,
                    "acc": acc,
                    "ppl": ppl,
                    "f1": f1,
                    "comp": comp,
                    "recon": recon,
                })
    except Exception as e:
        print(f"Error reading {p}: {e}")

print(f"TOTAL EVALUATIONS COLLECTED: {len(results)}\n")
print(f"{'Benchmark':<22} | {'Module':<22} | {'Rank':<8} | {'Status':<6} | {'Acc (%)':<8} | {'PPL':<8} | {'F1':<6} | {'Comp':<6} | {'Recon':<6}")
print("-" * 110)
for r in results:
    acc_str = f"{float(r['acc']):.2f}" if r['acc'] and r['acc'] != '0.0' else r['acc']
    ppl_str = f"{float(r['ppl']):.2f}" if r['ppl'] and r['ppl'] != '0.0' else r['ppl']
    f1_str = f"{float(r['f1']):.4f}" if r['f1'] and r['f1'] != '0.0' else r['f1']
    comp_str = f"{float(r['comp']):.2f}x" if r['comp'] and r['comp'] != '1.0' else r['comp']
    recon_str = f"{float(r['recon']):.4f}" if r['recon'] else ""
    print(f"{r['bench']:<22} | {r['module']:<22} | {r['rank']:<8} | {r['status']:<6} | {acc_str:<8} | {ppl_str:<8} | {f1_str:<6} | {comp_str:<6} | {recon_str:<6}")
