import glob
import csv
from pathlib import Path

BASE = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(".")

def extract_bench(bench_name):
    pattern = str(BASE / "results" / bench_name / "*" / "*" / "summary.csv")
    files = sorted(glob.glob(pattern))
    records = []
    for f in files:
        parts = Path(f).parts
        mod, r = parts[-3], parts[-2]
        rank_num = int(r.replace("rank_", "")) if "rank_" in r else 0
        try:
            with open(f, encoding="utf-8") as csvf:
                rows = list(csv.DictReader(csvf))
                if rows:
                    row = rows[0]
                    records.append({
                        "module": mod,
                        "rank": rank_num,
                        "status": row.get("status", ""),
                        "acc": float(row.get("accuracy_pct", 0) or 0),
                        "ppl": float(row.get("perplexity", 0) or 0),
                        "comp": float(row.get("compression_rate", 1) or 1),
                        "recon": float(row.get("avg_reconstruction_error", 0) or 0) if row.get("avg_reconstruction_error") else None,
                    })
        except Exception:
            pass
    return sorted(records, key=lambda x: (x["module"], x["rank"]))

print("Generating markdown tables...\n")

for bench in ["rank_sweep_sst2", "rank_sweep_roberta", "rank_sweep_gpt2"]:
    recs = extract_bench(bench)
    print(f"### {bench}")
    if bench == "rank_sweep_gpt2":
        print("| Module | Rank | Perplexity | Compression | Recon Error | Status |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for r in recs:
            ppl_str = f"{r['ppl']:.2f}" if r['ppl'] > 0 else "—"
            comp_str = f"{r['comp']:.2f}×" if r['comp'] > 1.0 else "1.00×"
            recon_str = f"{r['recon']:.4f}" if r['recon'] is not None else "—"
            print(f"| `{r['module']}` | {r['rank']} | {ppl_str} | {comp_str} | {recon_str} | **{r['status']}** |")
    else:
        print("| Module | Rank | Accuracy (%) | Compression | Recon Error | Status |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for r in recs:
            acc_str = f"{r['acc']:.2f}%" if r['acc'] > 0 else "—"
            comp_str = f"{r['comp']:.2f}×" if r['comp'] > 1.0 else "1.00×"
            recon_str = f"{r['recon']:.4f}" if r['recon'] is not None else "—"
            print(f"| `{r['module']}` | {r['rank']} | {acc_str} | {comp_str} | {recon_str} | **{r['status']}** |")
    print("\n")
