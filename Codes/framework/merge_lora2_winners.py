#!/usr/bin/env python3
"""
merge_lora2_winners.py
Merges lora_rank=2 rescue results into the main summary CSV.
Only overwrites REJECTED_EXPANSION rows that became PASS/PASS_LOW in the rescue run.

Usage:
    python merge_lora2_winners.py \
        --main_csv results/unified_sweep/distilbert_mnli_full/summary.csv \
        --rescue_csv results/unified_sweep/distilbert_mnli_full_lora2/summary.csv \
        --output_csv results/unified_sweep/distilbert_mnli_full/summary.csv
"""
import argparse
import pandas as pd
import shutil
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_csv", required=True)
    ap.add_argument("--rescue_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--dry_run", action="store_true", help="Print changes without writing")
    args = ap.parse_args()

    # Load both
    main_df = pd.read_csv(args.main_csv)
    rescue_df = pd.read_csv(args.rescue_csv)

    # Filter rescue to winners only (PASS or PASS_LOW)
    winners = rescue_df[rescue_df["status"].isin(["PASS", "PASS_LOW"])].copy()
    winners["lora_rank_used"] = 2  # annotate

    print(f"Main CSV: {len(main_df)} rows")
    print(f"Rescue CSV: {len(rescue_df)} rows")
    print(f"Winners (rescued): {len(winners)} modules")
    if len(winners) > 0:
        print("  " + ", ".join(sorted(winners["module_name"].tolist())))

    # Find main rows that were REJECTED_EXPANSION and are now rescued
    main_rejected = main_df[main_df["status"] == "REJECTED_EXPANSION"]["module_name"].tolist()
    rescued_names = [n for n in winners["module_name"].tolist() if n in main_rejected]
    print(f"\nModules changing REJECTED_EXPANSION → PASS/PASS_LOW: {len(rescued_names)}")
    for name in sorted(rescued_names):
        row = winners[winners["module_name"] == name].iloc[0]
        print(f"  {name}: acc={row.get('accuracy_pct', '?')} compression={row.get('compression_rate', '?')}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Backup main CSV
    backup_path = args.main_csv + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(args.main_csv, backup_path)
    print(f"\nBackup saved: {backup_path}")

    # Overwrite rows
    merged = main_df.copy()
    for name in rescued_names:
        rescue_row = winners[winners["module_name"] == name].iloc[0]
        idx = merged[merged["module_name"] == name].index
        if len(idx) > 0:
            for col in rescue_row.index:
                if col in merged.columns:
                    merged.loc[idx[0], col] = rescue_row[col]
            # Add lora_rank column if not present
            if "lora_rank_used" not in merged.columns:
                merged["lora_rank_used"] = ""
            merged.loc[idx[0], "lora_rank_used"] = 2

    # Also append any rescued modules that weren't in the main CSV at all
    main_names = merged["module_name"].tolist()
    new_modules = [n for n in winners["module_name"].tolist() if n not in main_names]
    if new_modules:
        new_rows = winners[winners["module_name"].isin(new_modules)].copy()
        if "lora_rank_used" not in merged.columns:
            merged["lora_rank_used"] = ""
        new_rows["lora_rank_used"] = 2
        merged = pd.concat([merged, new_rows], ignore_index=True)
        print(f"Appended {len(new_modules)} new modules: {new_modules}")

    merged.to_csv(args.output_csv, index=False)
    print(f"\n✅ Merged CSV written: {args.output_csv} ({len(merged)} rows)")
    print(f"   REJECTED_EXPANSION remaining: {len(merged[merged['status'] == 'REJECTED_EXPANSION'])}")
    print(f"   PASS/PASS_LOW: {len(merged[merged['status'].isin(['PASS', 'PASS_LOW'])])}")


if __name__ == "__main__":
    main()
