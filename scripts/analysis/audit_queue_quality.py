#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Deep forensic telemetry analysis of full_queue_all_modules/results.jsonl.
"""

import json
import sys
from pathlib import Path


def analyze_queue(results_path: str):
    """Execute analyze queue routine."""
    path = Path(results_path)
    if not path.exists():
        print(f"Error: {results_path} does not exist.")
        return

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    print("==================================================================")
    print(f"FORENSIC AUDIT OF BENCHMARK QUEUE ({total} total runs)")
    print("==================================================================")

    statuses = {}
    for r in records:
        s = r.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1
    print("\n1. STATUS BREAKDOWN:")
    for s, c in statuses.items():
        print(f"   - {s}: {c} ({c / total * 100:.1f}%)")

    types = {}
    for r in records:
        t = r.get("module_type", "unknown")
        types[t] = types.get(t, 0) + 1
    print("\n2. MODULE TYPE BREAKDOWN:")
    for t, c in types.items():
        print(f"   - {t}: {c}")

    valid_acc = [r for r in records if r.get("accuracy") is not None]
    valid_acc.sort(key=lambda x: (x.get("accuracy", 0), x.get("f1", 0)), reverse=True)

    print("\n3. TOP 15 PERFORMERS BY SST-2 ACCURACY & F1:")
    print(f"   {'Module':<32} {'Type':<12} {'Accuracy':<10} {'F1':<10} {'Loss':<10} {'VRAM (MB)':<10}")
    print(f"   {'-' * 32} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for r in valid_acc[:15]:
        vram = r.get("peak_vram_mb", 0.0) or 0.0
        print(
            f"   {r['module']:<32} {r.get('module_type', ''):<12} {r.get('accuracy', 0):<10.4f} {r.get('f1', 0):<10.4f} {r.get('loss', 0):<10.4f} {vram:<10.1f}"
        )

    nan_loss = [r for r in records if str(r.get("loss")).lower() == "nan"]
    inf_loss = [r for r in records if str(r.get("loss")).lower() == "inf"]
    low_acc = [r for r in records if r.get("accuracy") is not None and r.get("accuracy") < 0.50]
    high_loss = [r for r in records if r.get("loss") is not None and r.get("loss") > 5.0]

    print("\n4. NUMERICAL QUALITY & OUTLIERS:")
    print(f"   - NaN loss count: {len(nan_loss)}")
    print(f"   - Inf loss count: {len(inf_loss)}")
    print(f"   - Low accuracy (<0.50) count: {len(low_acc)}")
    print(f"   - High loss (>5.0) count: {len(high_loss)}")

    if low_acc:
        print("\n   Modules with Accuracy < 0.50:")
        for r in sorted(low_acc, key=lambda x: x.get("accuracy", 0)):
            print(
                f"     * {r['module']:<30} (type={r.get('module_type')}): acc={r.get('accuracy', 0):.4f}, f1={r.get('f1', 0):.4f}, loss={r.get('loss', 0):.4f}"
            )

    if high_loss:
        print("\n   Modules with Loss > 5.0:")
        for r in sorted(high_loss, key=lambda x: x.get("loss", 0), reverse=True):
            print(
                f"     * {r['module']:<30} (type={r.get('module_type')}): loss={r.get('loss', 0):.4f}, acc={r.get('accuracy', 0):.4f}"
            )

    comps = [r for r in records if r.get("module_type") == "compressor"]
    print(f"\n5. COMPRESSORS AUDIT ({len(comps)} compressors):")
    comps_sorted = sorted(comps, key=lambda x: (x.get("accuracy", 0), x.get("f1", 0)), reverse=True)
    print("   Top 5 Compressors:")
    for r in comps_sorted[:5]:
        print(
            f"     * {r['module']:<28}: acc={r.get('accuracy', 0):.4f}, f1={r.get('f1', 0):.4f}, loss={r.get('loss', 0):.4f}"
        )
    print("   Bottom 5 Compressors:")
    for r in comps_sorted[-5:]:
        print(
            f"     * {r['module']:<28}: acc={r.get('accuracy', 0):.4f}, f1={r.get('f1', 0):.4f}, loss={r.get('loss', 0):.4f}"
        )

    print("\nAudit complete.\n")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "results/full_queue_all_modules/results.jsonl"
    analyze_queue(p)
