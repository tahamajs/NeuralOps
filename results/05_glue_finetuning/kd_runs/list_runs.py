#!/usr/bin/env python3
"""List and summarize all KD runs from results/kd_runs/registry.json."""

import json
from pathlib import Path


def main():
    registry_file = Path("results") / "kd_runs" / "registry.json"
    if not registry_file.exists():
        print(f"Registry not found at {registry_file}.")
        return

    try:
        data = json.loads(registry_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error loading registry: {e}")
        return

    runs = data.get("runs", [])
    if not runs:
        print("No runs recorded in registry.")
        return

    print("=" * 105)
    print(
        f"{'Module':<20} | {'Timestamp':<16} | {'Post PPL':<10} | {'Recovery':<10} | {'Run Directory'}"
    )
    print("-" * 105)
    for r in runs:
        mod = r.get("module", "unknown")
        ts = r.get("timestamp", "unknown")
        post = r.get("best_post_ppl", float("nan"))
        rec = r.get("best_recovery_pct", 0.0)
        rdir = r.get("run_dir", "")
        post_str = f"{post:.2f}" if isinstance(post, (int, float)) and post < 1e9 else "N/A"
        rec_str = f"{rec:.2f}%" if isinstance(rec, (int, float)) else "0.00%"
        print(f"{mod:<20} | {ts:<16} | {post_str:<10} | {rec_str:<10} | {rdir}")
    print("=" * 105)


if __name__ == "__main__":
    main()
