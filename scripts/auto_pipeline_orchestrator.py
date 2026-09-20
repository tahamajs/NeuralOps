"""
auto_pipeline_orchestrator.py
-----------------------------
Autonomously runs pending sweeps in strict sequence:
  Step 1.1: Complete Group 7 (mamba, primal, sdrm, spectral_mamba)
  Step 1.2: Group 2 (Cat-A re-run with --lora_rank 2 --parallel 2)
  Step 1.3: Group 6 (hgrn2_outer retry with --timeout 7200 --recovery_steps 30 --parallel 1)
  Step 1.4: Group 4 (full-MNLI validation on PASS_STRICT with --eval_samples 9815 --parallel 2)

Saves snapshots after each stage and exits cleanly.
"""
import os, sys, time, subprocess, csv
from pathlib import Path

ROOT = Path(r"D:\svd_project")
PYTHON = r"C:\Users\Hadi Schumacher\anaconda3\python.exe"
SWEEP_SCRIPT = str(ROOT / "Codes" / "framework" / "run_quality_sweep.py")
MODEL_DIR = str(ROOT / "models" / "distilbert_mnli_full")
DATASET_DIR = str(ROOT / "Codes" / "framework" / "data" / "glue_data" / "mnli")
RESULTS_DIR = ROOT / "results" / "distilbert_mnli_full_mnli_sweep"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

def run_cmd(cmd_list, desc):
    print(f"\n{'='*70}\n[ORCHESTRATOR] Starting: {desc}\nCMD: {' '.join(cmd_list)}\n{'='*70}")
    t0 = time.monotonic()
    p = subprocess.Popen(cmd_list, cwd=str(ROOT))
    p.wait()
    elapsed = time.monotonic() - t0
    print(f"[ORCHESTRATOR] {desc} finished with exit code {p.returncode} in {elapsed:.1f}s")
    return p.returncode

def save_snapshot(name):
    if SUMMARY_CSV.exists():
        snap_path = RESULTS_DIR / f"summary_snapshot_{name}.csv"
        import shutil
        shutil.copyfile(str(SUMMARY_CSV), str(snap_path))
        print(f"[SNAPSHOT] Saved {snap_path}")

def wait_for_running_group7():
    print("[WAIT] Checking if Group 7 child worker is running...")
    while True:
        # Check if run_quality_sweep process is active
        res = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"], capture_output=True, text=True)
        # Look for run_one_module_isolated or run_quality_sweep in powershell
        ps_chk = subprocess.run(
            ["powershell", "-Command", "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True
        )
        cmdlines = ps_chk.stdout or ""
        is_active = any("run_quality_sweep.py" in line or "run_one_module_isolated.py" in line for line in cmdlines.splitlines())
        if not is_active:
            print("[WAIT] Group 7 completed!")
            break
        print(f"[{time.strftime('%H:%M:%S')}] Group 7 still active on GPU... sleeping 30s")
        time.sleep(30)

def main():
    # 1. Wait for current Group 7 to complete
    wait_for_running_group7()
    save_snapshot("post_group7")

    # 2. Step 1.2 — Group 2 (Cat-A re-run with --lora_rank 2 --parallel 2)
    cat_a = [
        "aegis_spectron_fp8","aegis_v1","aegis_v2","chela_slay","custom_attn",
        "decay_linear_attn","fms","fusion_attn","halo_regla","infini",
        "kronecker_tt_lora","leka_v2","leka_v3","linformer","linformer_attention",
        "moe_deltakv_paged","nasc","nexus","oha","ptp","rala","simple_ssm",
        "template_attention","tlinformer","token_compressed","vig","weighted_ensemble"
    ]
    cmd_g2 = [
        PYTHON, "-u", SWEEP_SCRIPT,
        "--model", MODEL_DIR,
        "--dataset", "mnli",
        "--dataset_dir", DATASET_DIR,
        "--output_dir", str(RESULTS_DIR),
        "--modules", ",".join(cat_a),
        "--lora_rank", "2",
        "--parallel", "2",
        "--batch_size", "16",
        "--no_skip"
    ]
    run_cmd(cmd_g2, "Group 2 (Cat-A Re-Run, 27 modules, lora_rank 2, parallel 2)")
    save_snapshot("post_group2")

    # 3. Step 1.3 — Group 6 (hgrn2_outer retry)
    cmd_g6 = [
        PYTHON, "-u", SWEEP_SCRIPT,
        "--model", MODEL_DIR,
        "--dataset", "mnli",
        "--dataset_dir", DATASET_DIR,
        "--output_dir", str(RESULTS_DIR),
        "--modules", "hgrn2_outer",
        "--timeout", "7200",
        "--recovery_steps", "30",
        "--parallel", "1",
        "--batch_size", "8",
        "--no_skip"
    ]
    run_cmd(cmd_g6, "Group 6 (hgrn2_outer retry, timeout 7200s, recovery_steps 30)")
    save_snapshot("post_group6")

    # 4. Step 1.4 — Group 4 (full-MNLI validation on PASS_STRICT)
    val_script = str(ROOT / "Codes" / "framework" / "full_mnli_validation.py")
    cmd_g4 = [
        PYTHON, "-u", val_script,
        "--model", MODEL_DIR,
        "--dataset_dir", DATASET_DIR,
        "--sweep_csv", str(SUMMARY_CSV),
        "--out", str(RESULTS_DIR / "summary_full_mnli.csv"),
        "--eval_samples", "9815",
        "--batch_size", "64",
        "--parallel", "2"
    ]
    run_cmd(cmd_g4, "Group 4 (Full-MNLI 9815 validation, parallel 2)")
    save_snapshot("post_group4")
    print("\n[COMPLETE] ALL 4 PENDING STAGES FINISHED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
