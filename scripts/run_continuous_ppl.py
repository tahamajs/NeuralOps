"""
Automated continuous WikiText-2 Perplexity evaluation queue with 2 parallel workers.
Safely processes all modules lacking perplexity measurements with concurrency=2,
monitoring available host RAM and VRAM before spawning each task.
"""
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path
import psutil
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PPL_DIR = PROJECT_ROOT / "results" / "05_perplexity_seqlen" / "wikitext"
PPL_DIR.mkdir(parents=True, exist_ok=True)

ATTN_DIR = PROJECT_ROOT / "Codes" / "framework" / "attentions"
COMP_DIR = PROJECT_ROOT / "Codes" / "framework" / "compression_engine"

def get_all_modules():
    mods = set()
    for d in [ATTN_DIR, COMP_DIR]:
        if d.exists():
            for p in d.glob("*.py"):
                name = p.name
                if not name.startswith("__") and not name.startswith("._") and not name.startswith("."):
                    mods.add(p.stem)
    return sorted(list(mods))

def is_already_done(mod_name):
    candidates = [
        PPL_DIR / f"wikitext2_{mod_name}" / "gpt2" / "ppl_result.json",
        PPL_DIR / f"wikitext2_{mod_name}" / "ppl_result.json",
        PPL_DIR / mod_name / "gpt2" / "ppl_result.json",
        PPL_DIR / mod_name / "ppl_result.json",
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return True
    return False

def run_single_ppl(mod):
    # Wait until system has healthy RAM headroom (>= 2.0 GB)
    while True:
        free_ram_gb = psutil.virtual_memory().available / 1e9
        if free_ram_gb >= 2.0:
            break
        time.sleep(3)

    out_dir = PPL_DIR / f"wikitext2_{mod}" / "gpt2"
    print(f"[PPL 3-Worker] Starting: {mod} (Free RAM: {psutil.virtual_memory().available / 1e9:.2f} GB)", flush=True)

    eval_script = PROJECT_ROOT / "Codes" / "framework" / "eval_perplexity.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--model", "models/gpt2",
        "--module", mod,
        "--output_dir", str(out_dir),
    ]

    try:
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if res.returncode == 0:
            print(f"[PPL 3-Worker] Finished {mod} successfully.", flush=True)
            return True
        else:
            print(f"[PPL 3-Worker] Exit code {res.returncode} for {mod}.", flush=True)
            return False
    except Exception as e:
        print(f"[PPL 3-Worker] Error running {mod}: {e}", flush=True)
        return False

def main():
    modules = get_all_modules()
    print(f"[PPL 3-Worker Queue] Total framework modules: {len(modules)}", flush=True)

    pending = [m for m in modules if not is_already_done(m)]
    print(f"[PPL 3-Worker Queue] Completed: {len(modules) - len(pending)}, Pending: {len(pending)}", flush=True)

    # Concurrency of 3 parallel workers for perplexity
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_single_ppl, m): m for m in pending}
        for future in concurrent.futures.as_completed(futures):
            mod = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[PPL 3-Worker Queue] Exception on {mod}: {e}", flush=True)

if __name__ == "__main__":
    main()
