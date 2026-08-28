import csv
from pathlib import Path

csv_path = Path("results/final_paper_run/summary.csv")
if csv_path.exists():
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    if reader:
        headers = list(reader[0].keys())
        # Keep only PASS rows, remove FAIL/TIMEOUT rows so they get retried with 900s
        clean_rows = [row for row in reader if row.get("status") == "PASS"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(clean_rows)
        print(f"Kept {len(clean_rows)} PASS rows, pruned failed/timed-out rows for 900s retry.")
