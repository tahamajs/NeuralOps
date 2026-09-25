import glob
import csv
from pathlib import Path

BASE = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(".")

for bench in ['rank_sweep_sst2', 'rank_sweep_roberta', 'rank_sweep_gpt2']:
    print(f"=== {bench} ===")
    pattern = str(BASE / "results" / bench / "*" / "*" / "summary.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print("  No summaries found.")
        continue
    for f in files:
        parts = Path(f).parts
        mod, r = parts[-3], parts[-2]
        with open(f, encoding="utf-8") as csvf:
            rows = list(csv.DictReader(csvf))
            if rows:
                row = rows[0]
                st = row.get('status', 'UNK')
                acc = row.get('accuracy_pct', '-')
                ppl = row.get('perplexity', '-')
                comp = row.get('compression_rate', '-')
                print(f"  {mod:<22} {r:<10} status={st:<20} acc={acc:<8} ppl={ppl:<8} comp={comp:<6}")
            else:
                print(f"  {mod:<22} {r:<10} EMPTY FILE")
