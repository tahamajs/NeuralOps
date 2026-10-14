#!/usr/bin/env python3
"""Deep forensic telemetry analysis of full_queue_all_modules/results.jsonl."""

import json
import sys
from pathlib import Path

def analyze_queue(results_path: str):
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
    print(f"==================================================================")
    print(f"FORENSIC AUDIT OF BENCHMARK QUEUE ({total} total runs)")
    print(f"==================================================================")

    # 1. Status breakdown
    statuses = {}
    for r in records:
        s = r.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1
    print(f"\n1. STATUS BREAKDOWN:")
    for s, c in statuses.items():
        print(f"   - {s}: {c} ({c/total*100:.1f}%)")

    # 2. Module types
    types = {}
    for r in records:
        t = r.get("module_type", "unknown")
        types[t] = types.get(t, 0) + 1
    print(f"\n2. MODULE TYPE BREAKDOWN:")
    for t, c in types.items():
        print(f"   - {t}: {c}")

    # 3. Top performers (Accuracy & F1)
    valid_acc = [r for r in records if r.get("accuracy") is not None]
    valid_acc.sort(key=lambda x: (x.get("accuracy", 0), x.get("f1", 0)), reverse=True)

    print(f"\n3. TOP 15 PERFORMERS BY SST-2 ACCURACY & F1:")
    print(f"   {'Module':<32} {'Type':<12} {'Accuracy':<10} {'F1':<10} {'Loss':<10} {'VRAM (MB)':<10}")
    print(f"   {'-'*32} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in valid_acc[:15]:
        vram = r.get("peak_vram_mb", 0.0) or 0.0
        print(f"   {r['module']:<32} {r.get('module_type',''):<12} {r.get('accuracy', 0):<10.4f} {r.get('f1', 0):<10.4f} {r.get('loss', 0):<10.4f} {vram:<10.1f}")

    # 4. Outliers & Degradations
    nan_loss = [r for r in records if str(r.get("loss")).lower() == "nan"]
    inf_loss = [r for r in records if str(r.get("loss")).lower() == "inf"]
    low_acc = [r for r in records if r.get("accuracy") is not None and r.get("accuracy") < 0.50]
    high_loss = [r for r in records if r.get("loss") is not None and r.get("loss") > 5.0]

    print(f"\n4. NUMERICAL QUALITY & OUTLIERS:")
    print(f"   - NaN loss count: {len(nan_loss)}")
    print(f"   - Inf loss count: {len(inf_loss)}")
    print(f"   - Low accuracy (<0.50) count: {len(low_acc)}")
    print(f"   - High loss (>5.0) count: {len(high_loss)}")

    if low_acc:
        print(f"\n   Modules with Accuracy < 0.50:")
        for r in sorted(low_acc, key=lambda x: x.get("accuracy", 0)):
            print(f"     * {r['module']:<30} (type={r.get('module_type')}): acc={r.get('accuracy',0):.4f}, f1={r.get('f1',0):.4f}, loss={r.get('loss',0):.4f}")

    if high_loss:
        print(f"\n   Modules with Loss > 5.0:")
        for r in sorted(high_loss, key=lambda x: x.get("loss", 0), reverse=True):
            print(f"     * {r['module']:<30} (type={r.get('module_type')}): loss={r.get('loss',0):.4f}, acc={r.get('accuracy',0):.4f}")

    # 5. Compressors audit
    comps = [r for r in records if r.get("module_type") == "compressor"]
    print(f"\n5. COMPRESSORS AUDIT ({len(comps)} compressors):")
    comps_sorted = sorted(comps, key=lambda x: (x.get("accuracy", 0), x.get("f1", 0)), reverse=True)
    print(f"   Top 5 Compressors:")
    for r in comps_sorted[:5]:
        print(f"     * {r['module']:<28}: acc={r.get('accuracy',0):.4f}, f1={r.get('f1',0):.4f}, loss={r.get('loss',0):.4f}")
    print(f"   Bottom 5 Compressors:")
    for r in comps_sorted[-5:]:
        print(f"     * {r['module']:<28}: acc={r.get('accuracy',0):.4f}, f1={r.get('f1',0):.4f}, loss={r.get('loss',0):.4f}")

    print(f"\nAudit complete.\n")

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "results/full_queue_all_modules/results.jsonl"
    analyze_queue(p)
