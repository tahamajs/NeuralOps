#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
registry_path = HERE / "results" / "kd_runs" / "registry.json"
unified_out = HERE / "results" / "01_unified_runs"

def main():
    if not registry_path.exists():
        print("No KD registry found. Ensure KD sweep is finished first.")
        sys.exit(1)

    with open(registry_path, "r", encoding="utf-8") as f:
        reg = json.load(f)

    passed_modules = []
    for r in reg.get('runs', []):
        mod = r.get('module')
        if not mod: continue
        res = r.get('results', {}).get('C', {})
        if res.get('status') == 'PASS':
            passed_modules.append(mod)

    print(f"Found {len(passed_modules)} PASSED modules eligible for Unified Runs.")

    orchestrator_script = HERE / "neuralops" / "runners" / "unified_orchestrator.py"
    
    for mod in passed_modules:
        expected_out = unified_out / f"unified_{mod}"
        if expected_out.exists() and len(list(expected_out.glob("*"))) > 0:
            print(f"Skipping {mod} (Unified run output already exists)")
            continue
        
        print("\n" + "="*80)
        print(f"LAUNCHING UNIFIED 4-MODEL SWEEP: {mod.upper()}")
        print("="*80)
        
        cmd = [sys.executable, str(orchestrator_script), mod]
        # Append any extra args from sys.argv (like --batch_size 16)
        if len(sys.argv) > 1:
            cmd.extend(sys.argv[1:])
            
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"ERROR: Unified run for {mod} failed! Skipping to next...")

if __name__ == "__main__":
    main()
