import csv
import json
import os
from pathlib import Path


def main():
    csv_path = Path("results/unified_sweep/distilbert_mnli_full/summary.csv")
    if not csv_path.exists():
        print("summary.csv not found")
        return

    rows = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            rows[r["module_name"]] = r

    isolated_dir = Path("logs/isolated_results")
    if not isolated_dir.exists():
        print("logs/isolated_results not found")
        return

    updated = 0
    # Process json files in chronological order so newest runs take precedence
    json_files = sorted(isolated_dir.glob("*.json"), key=os.path.getmtime)
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            mod = data.get("module")
            if not mod or mod not in rows:
                continue

            status = data.get("status", "UNKNOWN")
            rows[mod]["status"] = status
            if "accuracy_pct" in data or "accuracy" in data:
                rows[mod]["accuracy_pct"] = data.get("accuracy_pct", data.get("accuracy"))
            if "f1_score" in data or "f1" in data:
                rows[mod]["f1_score"] = data.get("f1_score", data.get("f1"))
            if "f1_macro" in data:
                rows[mod]["f1_macro"] = data.get("f1_macro")
            if "precision" in data:
                rows[mod]["precision"] = data.get("precision")
            if "recall" in data:
                rows[mod]["recall"] = data.get("recall")
            if "val_loss" in data:
                rows[mod]["val_loss"] = data.get("val_loss")
            if "peak_vram_mb" in data:
                rows[mod]["peak_vram_mb"] = data.get("peak_vram_mb")
            if "tokens_per_sec" in data:
                rows[mod]["tokens_per_sec"] = data.get("tokens_per_sec")
            if "total_params_before" in data or "params_before" in data:
                rows[mod]["total_params_before"] = data.get("total_params_before", data.get("params_before"))
            if "total_params_after" in data or "params_after" in data:
                rows[mod]["total_params_after"] = data.get("total_params_after", data.get("params_after"))
            if "trainable_recovery_params" in data:
                rows[mod]["trainable_recovery_params"] = data.get("trainable_recovery_params")
            if "compression_rate" in data or "compression_ratio" in data:
                rows[mod]["compression_rate"] = data.get("compression_rate", data.get("compression_ratio"))
            if "error" in data:
                rows[mod]["error"] = data.get("error")
            if "finished_at" in data:
                rows[mod]["timestamp"] = data.get("finished_at")
            updated += 1
        except Exception as e:
            pass

    tmp = csv_path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in rows:
            writer.writerow(rows[m])

    if tmp.exists():
        csv_path.unlink()
        tmp.rename(csv_path)

    print(f"Sync complete: checked {len(json_files)} results, updated summary.csv")


if __name__ == "__main__":
    main()
