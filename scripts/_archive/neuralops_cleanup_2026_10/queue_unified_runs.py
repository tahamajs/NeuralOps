#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
registry_path = HERE / "results" / "kd_runs" / "registry.json"
unified_out = HERE / "results" / "01_unified_runs"

def get_passed_modules():
    if not registry_path.exists(): return []
    try:
        with open(registry_path, "r", encoding="utf-8") as f: reg = json.load(f)
    except: return []
    
    passed = []
    for r in reg.get('runs', []):
        mod = r.get('module')
        if not mod: continue
        res = r.get('results', {}).get('C', {})
        if res.get('status') == 'PASS':
            passed.append(mod)
    return passed

def main():
    print("Starting Continuous Unified Pipeline Monitor...")
    orchestrator_script = HERE / "neuralops" / "runners" / "unified_orchestrator.py"
    processed = set()
    
    while True:
        passed = get_passed_modules()
        new_modules = [m for m in passed if m not in processed]
        
        for mod in new_modules:
            expected_out = unified_out / f"unified_{mod}"
            if expected_out.exists() and len(list(expected_out.glob("*"))) > 0:
                print(f"Skipping {mod} (Unified run output already exists)")
                processed.add(mod)
                continue
                
            print("\n" + "="*80)
            print(f"LAUNCHING UNIFIED 4-MODEL SWEEP: {mod.upper()}")
            print("="*80)
            
            cmd = [sys.executable, str(orchestrator_script), mod]
            if len(sys.argv) > 1: cmd.extend(sys.argv[1:])
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"ERROR: Unified run for {mod} failed!")
            finally:
                processed.add(mod)
                
        # Sleep and wait for KD sweep to spit out more modules
        time.sleep(30)

if __name__ == "__main__":
    main()
