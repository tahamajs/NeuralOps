import subprocess
import sys
import time
import os

# Priority queue of hardened modules to run across RoBERTa and GPT-2 (and Qwen3.5 where supported)
<<<<<<< HEAD
# Excluded failed modules: mamba, svd_mamba, aegis_hybrid
QUEUE = [
    ("svd_compressed", ["distilbert", "roberta", "gpt2"]),
=======
QUEUE = [
    ("svd_compressed", ["distilbert", "roberta", "gpt2"]),
    ("svd_mamba", ["distilbert", "roberta", "gpt2"]),
>>>>>>> 69da8ec1a (add files of results)
    ("nexus", ["distilbert", "roberta", "gpt2"]),
    ("xattention", ["distilbert", "roberta", "gpt2"]),
    ("soa_module", ["distilbert", "roberta", "gpt2"]),
    ("reap_moe", ["distilbert", "roberta", "gpt2"]),
    ("regla_attention", ["distilbert", "roberta", "gpt2"]),
    ("phantom", ["distilbert", "roberta", "gpt2"]),
    ("vspec", ["distilbert", "roberta", "gpt2"]),
    ("orion", ["distilbert", "roberta", "gpt2"]),
    ("rala", ["distilbert", "roberta", "gpt2"]),
    ("tpa", ["distilbert", "roberta", "gpt2"]),
]

SCRIPT_MAP = {
    "svd_compressed": "run_all_4_svd_compressed.py",
    "svd_mamba": "run_all_4_svd_mamba.py",
    "nexus": "run_all_4_nexus.py",
    "xattention": "run_all_4_xattention.py",
    "soa_module": "run_all_4_soa_module.py",
    "reap_moe": "run_all_4_reap_moe.py",
    "regla_attention": "run_all_4_regla.py",
    "phantom": "run_all_4_phantom.py",
    "vspec": "run_all_4_vspec.py",
    "orion": "run_all_4_orion.py",
    "rala": "run_all_4_rala.py",
    "tpa": "run_all_4_tpa.py",
}

def main():
    if os.path.exists(r"D:\svd_project"):
        os.chdir(r"D:\svd_project")
    
    print("=" * 80)
    print("STARTING SEQUENTIAL EVALUATION OF ALL HARDENED ATTENTION MODULES")
    print("=" * 80)
    
    for mod_name, models_to_run in QUEUE:
        script = SCRIPT_MAP.get(mod_name)
        if not script or not os.path.exists(script):
            print(f"[-] Script not found for {mod_name}: {script}")
            continue
        
        models_arg = ",".join(models_to_run)
        cmd = [sys.executable, script, "--models", models_arg]
        print(f"\n>>> [LAUNCHING] {mod_name.upper()} on models: {models_arg}")
        print(f"Command: {' '.join(cmd)}")
        
        t0 = time.time()
        res = subprocess.run(cmd)
        elapsed = time.time() - t0
        status_str = "SUCCESS" if res.returncode == 0 else f"FAILED (code {res.returncode})"
        print(f"<<< [FINISHED] {mod_name.upper()} in {elapsed:.1f}s | Result: {status_str}\n")
        time.sleep(2)

if __name__ == "__main__":
    main()
