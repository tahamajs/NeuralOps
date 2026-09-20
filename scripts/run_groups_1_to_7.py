"""
run_groups_1_to_7.py  — executes all pending groups sequentially on hadi-windows
Writes logs to D:\svd_project\logs\pending_<group>_<timestamp>.log
"""
from __future__ import annotations
import csv, json, math, os, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

BASE   = Path("D:/svd_project")
RESULT = BASE / "results/distilbert_mnli_full_mnli_sweep"
LOGS   = BASE / "logs"
LOGS.mkdir(exist_ok=True)
MODEL  = BASE / "models/distilbert_mnli_full"
DS     = BASE / "Codes/framework/data/glue_data/mnli"
SWEEP  = BASE / "Codes/framework/run_quality_sweep.py"
AUDIT  = BASE / "Codes/framework/per_class_collapse_audit.py"
FULL_V = BASE / "Codes/framework/full_mnli_validation.py"
SUMMARY= RESULT / "summary.csv"

STOP = False   # global stop flag

def log(msg: str, fp=None):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if fp:
        fp.write(line + "\n"); fp.flush()

def gpu_check(fp=None) -> bool:
    """Returns True if GPU is safe to proceed."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True)
    if r.returncode != 0:
        log("WARNING: nvidia-smi failed", fp); return True  # assume OK
    temp_str, vram_str = r.stdout.strip().split(",")
    temp = int(temp_str.strip())
    vram = int(re.search(r'\d+', vram_str).group())
    log(f"GPU: temp={temp}°C  VRAM={vram}MiB", fp)
    if temp > 90:
        log(f"STOP: GPU temp {temp}°C > 90°C", fp); return False
    if vram > 7500:
        log(f"STOP: VRAM {vram}MiB > 7500MiB", fp); return False
    return True

def run_sweep(modules: list[str], extra_args: list[str], tag: str, fp=None) -> dict:
    """Runs run_quality_sweep.py for a module list. Returns result dict."""
    log(f"  Modules ({len(modules)}): {', '.join(modules[:8])}{'...' if len(modules)>8 else ''}", fp)
    cmd = [
        sys.executable, str(SWEEP),
        "--model", str(MODEL),
        "--dataset", "mnli",
        "--dataset_dir", str(DS),
        "--modules", *modules,
    ] + extra_args
    log(f"  CMD: python run_quality_sweep.py --modules <{len(modules)}> {' '.join(extra_args)}", fp)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True)
    elapsed = round(time.monotonic() - t0, 1)
    log(f"  Exit code: {proc.returncode}  Elapsed: {elapsed}s", fp)
    if proc.stdout: fp.write(proc.stdout[-3000:]); fp.flush()
    if proc.stderr: fp.write(proc.stderr[-3000:]); fp.flush()
    return {"returncode": proc.returncode, "elapsed": elapsed, "stdout": proc.stdout[-500:]}

def read_summary_rows(mods: list[str]) -> list[dict]:
    """Read the latest rows for given modules from summary.csv."""
    with open(SUMMARY) as f:
        all_rows = {r["module_name"]: r for r in csv.DictReader(f)}
    return [all_rows[m] for m in mods if m in all_rows]

def print_group_report(mods: list[str], tag: str, fp=None):
    rows = read_summary_rows(mods)
    passes = sum(1 for r in rows if r.get("status","") == "PASS")
    lows   = sum(1 for r in rows if r.get("status","") == "PASS_LOW")
    fails  = sum(1 for r in rows if r.get("status","") in ("FAIL","EVAL_FAILED"))
    exp    = sum(1 for r in rows if "EXPANSION" in r.get("status",""))
    log(f"\n  === {tag} REPORT: {len(rows)}/{len(mods)} rows found ===", fp)
    log(f"  PASS={passes}  PASS_LOW={lows}  FAIL={fails}  EXPANSION={exp}", fp)
    log(f"  Threshold used (from error col): ", fp)
    for r in rows:
        err = r.get("error","")
        m = re.search(r'threshold \((\d+\.\d+)%\)', err)
        thr = m.group(1) if m else "N/A (above threshold)"
        log(f"    {r['module_name']:35} acc={float(r.get('accuracy_pct',0)):5.1f}%  status={r['status']:22}  thr={thr}", fp)
    if passes == 0 and len(rows) > 0:
        log("  !!! ZERO PASS MODULES — STOP CONDITION MET !!!", fp)
        global STOP; STOP = True
    return rows

# ──────────────────────────────────────────────────────────────────────────────
# GROUP 5 — Collapse Audit
# ──────────────────────────────────────────────────────────────────────────────
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
G5_LOG = LOGS / f"pending_group5_collapse_{TS}.log"
with open(G5_LOG, "w") as fp:
    log("="*60, fp)
    log("GROUP 5: Per-class collapse audit", fp)
    log("="*60, fp)

    if not gpu_check(fp): sys.exit(1)

    AUDIT_OUT = RESULT / "collapse_audit.csv"
    SUSPECTS = ["galerkin_attention","kda_attention","simple_ssm",
                "mla_attention","hgrn2_layer","aegis_spectron_fp8","mamba"]
    cmd = [
        sys.executable, str(AUDIT),
        "--model", str(MODEL),
        "--dataset_dir", str(DS),
        "--sweep_csv", str(SUMMARY),
        "--modules", *SUSPECTS,
        "--out", str(AUDIT_OUT),
        "--batch_size", "64",
    ]
    log(f"CMD: {' '.join(cmd[:5])} ...", fp)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True)
    elapsed = round(time.monotonic() - t0, 1)
    fp.write(proc.stdout); fp.flush()
    fp.write(proc.stderr); fp.flush()
    log(f"Exit={proc.returncode}  Elapsed={elapsed}s", fp)

    if not AUDIT_OUT.exists():
        log("STOP: collapse_audit.csv not created. Aborting.", fp)
        sys.exit(2)

    with open(AUDIT_OUT) as f:
        audit_rows = list(csv.DictReader(f))
    log(f"collapse_audit.csv: {len(audit_rows)} rows", fp)
    log("\nFull collapse audit table:", fp)
    for r in audit_rows:
        log(f"  {r.get('module','?'):35}  collapse={r.get('collapse_confirmed','?'):5}  "
            f"dominant_class={r.get('dominant_class','?')}  "
            f"dominant_frac={r.get('dominant_frac','?')}  "
            f"audit_acc={r.get('audit_acc','?')}", fp)

    log("GROUP 5 DONE", fp)

print(f"\nGroup 5 log: {G5_LOG}")
print(f"collapse_audit.csv exists: {AUDIT_OUT.exists()}")
print(f"rows: {len(audit_rows) if 'audit_rows' in dir() else 'N/A'}")

# Print summary to stdout for SSH capture
if "audit_rows" in dir():
    for r in audit_rows:
        print(f"AUDIT|{r.get('module')}|collapse={r.get('collapse_confirmed')}|"
              f"dom_class={r.get('dominant_class')}|dom_frac={r.get('dominant_frac')}|"
              f"audit_acc={r.get('audit_acc')}")
