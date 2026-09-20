"""
run_all_pending_groups.py  — runs Groups 1, 2, 3, 6, 7 on hadi-windows
Must be run locally on hadi-windows:
  cd D:\svd_project
  python scripts\run_all_pending_groups.py > logs\pending_all_groups.log 2>&1

This script does NOT run Group 4 (full-MNLI validation) or Group 5 (collapse audit).
Those are separate scripts.
"""
from __future__ import annotations
import csv, json, math, os, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

BASE   = Path("D:/svd_project")
RESULT = BASE / "results/distilbert_mnli_full_mnli_sweep"
LOGS   = BASE / "logs"
LOGS.mkdir(exist_ok=True)
MODEL  = str(BASE / "models/distilbert_mnli_full")
DS     = str(BASE / "Codes/framework/data/glue_data/mnli")
SWEEP  = str(BASE / "Codes/framework/run_quality_sweep.py")
SUMMARY = RESULT / "summary.csv"

STOP_FLAG = [False]
GPU_STOP_TEMP  = 90
GPU_STOP_VRAM  = 7500

MASTER_LOG = LOGS / f"pending_all_groups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_mfp = open(MASTER_LOG, "w", encoding="utf-8")

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _mfp.write(line + "\n"); _mfp.flush()

def gpu_check() -> bool:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        log("WARNING: nvidia-smi failed — assuming GPU safe")
        return True
    parts = r.stdout.strip().split(",")
    temp = int(parts[0].strip())
    vram = int(re.search(r'\d+', parts[1]).group())
    util = parts[2].strip()
    log(f"GPU: temp={temp}°C  VRAM={vram}MiB  util={util}")
    if temp > GPU_STOP_TEMP:
        log(f"!!! STOP: GPU temp {temp}°C > {GPU_STOP_TEMP}°C !!!")
        STOP_FLAG[0] = True
        return False
    if vram > GPU_STOP_VRAM:
        log(f"!!! STOP: VRAM {vram}MiB > {GPU_STOP_VRAM}MiB !!!")
        STOP_FLAG[0] = True
        return False
    return True

def snapshot_summary(tag: str):
    dst = RESULT / f"summary_snapshot_{tag}.csv"
    if SUMMARY.exists():
        import shutil; shutil.copy(SUMMARY, dst)
        log(f"Snapshot saved: {dst.name}")
    else:
        log(f"WARNING: summary.csv not found, snapshot skipped")

def run_sweep_group(group_name: str, modules: list[str], extra_args: list[str],
                    out_dir_name: str) -> dict:
    """Run run_quality_sweep.py for the given module list. Returns summary dict."""
    log(f"\n{'='*60}")
    log(f"STARTING {group_name} ({len(modules)} modules)")
    log(f"{'='*60}")

    if not gpu_check():
        return {"status": "STOP", "reason": "GPU limit exceeded"}

    # Take pre-group snapshot
    snapshot_summary(f"pre_{group_name.replace(' ','_')}")

    out_dir = RESULT.parent / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, SWEEP,
        "--model", MODEL,
        "--dataset", "mnli",
        "--dataset_dir", DS,
        "--modules", *modules,
        "--output_dir", str(out_dir),
    ] + extra_args

    log(f"CMD: python run_quality_sweep.py "
        f"--modules <{len(modules)}> {' '.join(extra_args)} "
        f"--output_dir {out_dir_name}")

    t0 = time.monotonic()
    proc = subprocess.run(
        cmd, cwd=str(BASE),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace"
    )
    elapsed = round(time.monotonic() - t0, 1)

    # Write full stdout to master log
    _mfp.write(proc.stdout[-8000:]); _mfp.flush()

    log(f"Exit={proc.returncode}  Elapsed={elapsed}s")

    # Read results from the output dir summary.csv
    out_csv = out_dir / "summary.csv"
    if not out_csv.exists():
        # Fall back to main summary.csv for these modules
        out_csv = SUMMARY
        log(f"WARNING: {out_dir_name}/summary.csv not found, reading main summary.csv")

    rows_found = {}
    if out_csv.exists():
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                if r["module_name"] in modules:
                    rows_found[r["module_name"]] = r

    # Post-group snapshot
    snapshot_summary(f"post_{group_name.replace(' ','_')}")

    # Report
    passes = sum(1 for r in rows_found.values() if r.get("status","") == "PASS")
    lows   = sum(1 for r in rows_found.values() if r.get("status","") == "PASS_LOW")
    fails  = sum(1 for r in rows_found.values() if r.get("status","") in ("FAIL","EVAL_FAILED"))
    exp    = sum(1 for r in rows_found.values() if "EXPANSION" in r.get("status",""))
    missing = [m for m in modules if m not in rows_found]

    log(f"\n--- {group_name} RESULTS ---")
    log(f"Modules run: {len(modules)}, Found in CSV: {len(rows_found)}, Missing: {len(missing)}")
    log(f"PASS={passes}  PASS_LOW={lows}  FAIL={fails}  EXPANSION={exp}")
    if missing:
        log(f"MISSING from CSV: {', '.join(missing)}")
    for m, r in sorted(rows_found.items(), key=lambda x: -float(x[1].get("accuracy_pct",0) or 0)):
        err = r.get("error","")
        thr_m = re.search(r'threshold \((\d+\.\d+)%\)', err)
        thr = thr_m.group(1) if thr_m else "N/A"
        delta_m = re.search(r'\+(\d+\.\d+)%\)', err)
        delta = f"+{delta_m.group(1)}%" if delta_m else ""
        log(f"  {m:38} acc={float(r.get('accuracy_pct',0)):5.1f}%  "
            f"status={r.get('status','?'):25} thr={thr} {delta}")

    if passes == 0 and len(rows_found) > 0 and group_name not in ("Group 6 - hgrn2_outer",):
        log(f"!!! STOP: Zero PASS modules in {group_name} !!!")
        STOP_FLAG[0] = True

    return {
        "group": group_name, "modules": len(modules), "found": len(rows_found),
        "passes": passes, "lows": lows, "fails": fails, "expansion": exp,
        "elapsed": elapsed, "exit_code": proc.returncode, "missing": missing,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Read module lists from summary.csv
# ──────────────────────────────────────────────────────────────────────────────
log("Reading summary.csv to build module lists...")
with open(SUMMARY) as f:
    all_rows = {r["module_name"]: r for r in csv.DictReader(f)}

# GROUP 1: threshold flip (36.90 <= acc < 38.60)
GRP1 = [m for m,r in all_rows.items()
        if 36.90 <= float(r.get("accuracy_pct",0) or 0) < 38.60]

# GROUP 2: Cat A expansion (delta < 5%)
GRP2 = []
for m,r in all_rows.items():
    if r.get("status","") not in ("MODERATE_EXPANSION","EXTREME_EXPANSION"): continue
    err = r.get("error","")
    dm = re.search(r'\+(\d+\.\d+)%\)', err)
    if dm and float(dm.group(1)) < 5.0:
        GRP2.append(m)

# GROUP 3: Cat B/C expansion (delta >= 5%)
GRP3 = []
for m,r in all_rows.items():
    if r.get("status","") not in ("MODERATE_EXPANSION","EXTREME_EXPANSION"): continue
    err = r.get("error","")
    dm = re.search(r'\+(\d+\.\d+)%\)', err)
    if dm and float(dm.group(1)) >= 5.0:
        GRP3.append(m)

# GROUP 7: FAIL/PASS_LOW with acc < 36.90 (re-run with 100 steps)
GRP7 = [m for m,r in all_rows.items()
        if float(r.get("accuracy_pct",0) or 0) < 36.90
        and r.get("status","") in ("PASS_LOW", "FAIL", "EVAL_FAILED")]

log(f"Group 1: {len(GRP1)} modules: {', '.join(sorted(GRP1))}")
log(f"Group 2: {len(GRP2)} modules")
log(f"Group 3: {len(GRP3)} modules: {', '.join(sorted(GRP3))}")
log(f"Group 7: {len(GRP7)} modules: {', '.join(sorted(GRP7))}")

results_summary = []

# ──────────────────────────────────────────────────────────────────────────────
# GROUP 1 — Threshold-flip re-runs (~20 min)
# ──────────────────────────────────────────────────────────────────────────────
r1 = run_sweep_group(
    "Group 1 - Threshold Flip", GRP1,
    ["--lora_rank", "16", "--batch_size", "16", "--parallel", "2"],
    "distilbert_mnli_full_mnli_sweep_g1_thresholdfix"
)
results_summary.append(r1)
if STOP_FLAG[0]: 
    log("STOP triggered after Group 1. Halting."); sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# GROUP 3 — Cat B/C re-runs (smaller, do before Cat A for faster feedback)
# ──────────────────────────────────────────────────────────────────────────────
r3 = run_sweep_group(
    "Group 3 - Cat B/C Expansion", GRP3,
    ["--lora_rank", "4", "--batch_size", "16", "--parallel", "2"],
    "distilbert_mnli_full_mnli_sweep_g3_catBC"
)
results_summary.append(r3)
if STOP_FLAG[0]:
    log("STOP triggered after Group 3. Halting."); sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# GROUP 7 — FAIL re-runs with 100 recovery steps
# ──────────────────────────────────────────────────────────────────────────────
r7 = run_sweep_group(
    "Group 7 - FAIL Re-runs", GRP7,
    ["--lora_rank", "16", "--batch_size", "8", "--parallel", "2",
     "--recovery_steps", "100"],
    "distilbert_mnli_full_mnli_sweep_g7_fail_rerun"
)
results_summary.append(r7)
if STOP_FLAG[0]:
    log("STOP triggered after Group 7. Halting."); sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# GROUP 6 — hgrn2_outer retry with 7200s timeout, 30 recovery steps
# ──────────────────────────────────────────────────────────────────────────────
r6 = run_sweep_group(
    "Group 6 - hgrn2_outer", ["hgrn2_outer"],
    ["--lora_rank", "16", "--batch_size", "8", "--parallel", "1",
     "--timeout", "7200", "--recovery_steps", "30"],
    "distilbert_mnli_full_mnli_sweep_g6_hgrn2_outer"
)
results_summary.append(r6)
# NOTE: hgrn2_outer TIMEOUT is not a zero-PASS stop condition

# ──────────────────────────────────────────────────────────────────────────────
# GROUP 2 — Cat A re-runs (largest group, ~3.5 hr)
# ──────────────────────────────────────────────────────────────────────────────
r2 = run_sweep_group(
    "Group 2 - Cat A Expansion", GRP2,
    ["--lora_rank", "2", "--batch_size", "16", "--parallel", "3"],
    "distilbert_mnli_full_mnli_sweep_g2_catA"
)
results_summary.append(r2)
if STOP_FLAG[0]:
    log("STOP triggered after Group 2. Halting."); sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
log(f"\n{'='*60}")
log("ALL GROUPS COMPLETE")
log(f"{'='*60}")
for res in results_summary:
    g = res.get("group","?")
    log(f"  {g:40} PASS={res.get('passes',0):3}  "
        f"FAIL={res.get('fails',0):3}  "
        f"LOW={res.get('lows',0):3}  "
        f"EXPANSION={res.get('expansion',0):3}  "
        f"elapsed={res.get('elapsed',0):.0f}s  "
        f"exit={res.get('exit_code','?')}")

# Write sentinel
(LOGS / "all_groups_sentinel.txt").write_text("DONE\n")
log(f"Sentinel written. Master log: {MASTER_LOG}")
_mfp.close()
