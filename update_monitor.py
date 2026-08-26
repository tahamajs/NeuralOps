import re
from pathlib import Path

path = Path("Codes/framework/live_monitor.py")
content = path.read_text(encoding="utf-8")

# 1. Update nvsmi_candidates order
content = content.replace(
    '    nvsmi_candidates = [\n        "nvidia-smi",\n        r"C:\\Windows\\System32\\nvidia-smi.exe",',
    '    nvsmi_candidates = [\n        r"C:\\Windows\\System32\\nvidia-smi.exe",\n        "nvidia-smi",'
)

# 2. Update get_completed_results to check active summary.csv first
old_get_completed = '''def get_completed_results() -> List[Dict[str, Any]]:
    """Parses comprehensive_run_detached.log and returns structured results with error details."""
    results = []
    error_map = {}'''

new_get_completed = '''def get_completed_results() -> List[Dict[str, Any]]:
    """Parses active quality sweep summary.csv or comprehensive_run_detached.log."""
    results = []
    error_map = {}
    
    # Check active Phase 2 summary.csv first
    active_csv_candidates = [
        PROJECT_DIR / "results" / "run_quality_fixed_final" / "summary.csv",
        Path("results/run_quality_fixed_final/summary.csv"),
    ]
    for acsv in active_csv_candidates:
        if acsv.exists() and acsv.stat().st_size > 0:
            try:
                import csv
                with open(acsv, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        mod = row.get("module_name", row.get("module", ""))
                        if not mod:
                            continue
                        fwd = float(row["fwd_ms"]) if row.get("fwd_ms") and row["fwd_ms"] not in ("-", "") else None
                        bwd = float(row["bwd_ms"]) if row.get("bwd_ms") and row["bwd_ms"] not in ("-", "") else None
                        mem = float(row["peak_vram_mb"]) if row.get("peak_vram_mb") and row["peak_vram_mb"] not in ("-", "") else None
                        acc = float(row["mean_accuracy"]) if row.get("mean_accuracy") and row["mean_accuracy"] not in ("-", "") else None
                        f1 = float(row["f1_score"]) if row.get("f1_score") and row["f1_score"] not in ("-", "") else None
                        fid = float(row["fidelity_score"]) if row.get("fidelity_score") and row["fidelity_score"] not in ("-", "") else None
                        thru = float(row["throughput_tokens_sec"]) if row.get("throughput_tokens_sec") and row["throughput_tokens_sec"] not in ("-", "") else None
                        pval = float(row["p_value"]) if row.get("p_value") and row["p_value"] not in ("-", "") else None
                        ci_lower = float(row["ci_lower"]) if row.get("ci_lower") and row["ci_lower"] not in ("-", "") else None
                        ci_upper = float(row["ci_upper"]) if row.get("ci_upper") and row["ci_upper"] not in ("-", "") else None
                        
                        ts_val = row.get("timestamp", datetime.now().isoformat())
                        try:
                            ts = datetime.fromisoformat(ts_val)
                        except Exception:
                            ts = datetime.now()
                            
                        results.append({
                            "timestamp": ts,
                            "track": "quality_phase2",
                            "module": mod,
                            "model": row.get("model_family", "BERT").upper(),
                            "dataset": row.get("dataset", "SST2").upper(),
                            "status": row.get("status", "PASS").upper(),
                            "fwd_ms": fwd,
                            "bwd_ms": bwd,
                            "mem_mb": mem,
                            "acc_pct": acc,
                            "f1_score": f1,
                            "fidelity_score": fid,
                            "throughput": thru,
                            "p_value": pval,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "error_detail": "None / Success",
                        })
                if results:
                    return results
            except Exception:
                pass'''

content = content.replace(old_get_completed, new_get_completed, 1)

# 3. Update is_sweep_running
old_is_running = '''def is_sweep_running() -> bool:
    """Checks if the python benchmark launcher is actively running."""
    try:
        cmd = ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return bool(res.stdout.strip())
    except Exception:
        return False'''

new_is_running = '''def is_sweep_running() -> bool:
    """Checks if the python benchmark launcher is actively running."""
    try:
        cmd = ["powershell", "-Command", "Get-WmiObject Win32_Process -Filter \\\"name='python.exe'\\\" | Select-Object -ExpandProperty CommandLine"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            for line in res.stdout.strip().splitlines():
                if "run_quality_sweep.py" in line or "eval_all_models_datasets.py" in line:
                    return True
        return False
    except Exception:
        return False'''

content = content.replace(old_is_running, new_is_running, 1)

# 4. In _parse_single_run_dir, ignore run_quality_fixed_final if sweep is active
content = content.replace(
    '        run_id = dir_path.name\n        cfg_file = dir_path / "config.json"',
    '        run_id = dir_path.name\n        if run_id == "run_quality_fixed_final" and is_sweep_running():\n            return\n        cfg_file = dir_path / "config.json"'
)

# 5. In get_run_details, ensure gpu and progress are returned
content = content.replace(
    '            "counts": {\n                "pass": sum(1 for r in results if r.get("status") == "PASS"),\n                "fail": sum(1 for r in results if r.get("status") == "FAIL"),\n                "timeout": sum(1 for r in results if r.get("status") == "TIMEOUT"),\n                "total": len(results),\n            },\n            "results": results,\n        }',
    '            "counts": {\n                "pass": sum(1 for r in results if r.get("status") == "PASS"),\n                "fail": sum(1 for r in results if r.get("status") == "FAIL"),\n                "timeout": sum(1 for r in results if r.get("status") == "TIMEOUT"),\n                "total": len(results),\n            },\n            "results": results,\n            "gpu": get_gpu_stats(),\n            "progress": compute_eta_and_progress(results, total_combos=240),\n        }'
)

# 6. In list_runs, ensure active run is properly named Phase 2
content = content.replace(
    '"name": "Live Adaptive Sweep (v3 - Active)",',
    '"name": "Phase 2 Quality Sweep (Active)",'
)

path.write_text(content, encoding="utf-8")
print("Updated Codes/framework/live_monitor.py successfully.")
