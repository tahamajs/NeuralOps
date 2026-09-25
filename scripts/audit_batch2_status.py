import glob
import csv
from pathlib import Path

BASE = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(".")

print(f"{'Module':<24} | {'Rank':<8} | {'Model':<12} | {'Status':<18} | {'Acc (%)':<8} | {'Comp':<6} | {'Recon':<6}")
print("-" * 90)

for p in sorted(glob.glob(str(BASE / "results" / "batch2_*" / "*" / "*" / "summary.csv"))):
    parts = Path(p).parts
    bench = parts[-4].replace("batch2_", "")
    mod = parts[-3]
    r = parts[-2]
    try:
        with open(p, encoding="utf-8") as f:
            r_dict = list(csv.DictReader(f))[0]
            st = r_dict.get('status', 'UNK')
            acc = float(r_dict.get('accuracy_pct', 0) or 0)
            comp = float(r_dict.get('compression_rate', 1) or 1)
            recon = float(r_dict.get('avg_reconstruction_error', 0) or 0) if r_dict.get('avg_reconstruction_error') else None
            acc_str = f"{acc:.2f}%" if acc > 0 else "-"
            comp_str = f"{comp:.2f}x" if comp > 1.0 else "1.00x"
            recon_str = f"{recon:.4f}" if recon is not None else "-"
            print(f"{mod:<24} | {r:<8} | {bench:<12} | {st:<18} | {acc_str:<8} | {comp_str:<6} | {recon_str:<6}")
    except Exception as e:
        print(f"Error {p}: {e}")
