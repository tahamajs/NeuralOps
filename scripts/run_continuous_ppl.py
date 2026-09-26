"""
Automated continuous WikiText-2 Perplexity evaluation queue.
Iterates over all framework modules that lack a perplexity measurement
and computes PPL using standard WikiText-2 protocol with 1 worker to run
alongside the zero-shot evaluation without exceeding RAM or GPU limits.
"""
import subprocess
import sys
from pathlib import Path

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
    # Check possible result directory naming conventions
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

def main():
    modules = get_all_modules()
    print(f"[PPL Queue] Found {len(modules)} total framework modules.", flush=True)
    
    pending = [m for m in modules if not is_already_done(m)]
    print(f"[PPL Queue] {len(modules) - len(pending)} already completed. {len(pending)} pending.", flush=True)
    
    python_bin = sys.executable
    eval_script = PROJECT_ROOT / "Codes" / "framework" / "eval_perplexity.py"
    
    for idx, mod in enumerate(pending, 1):
        out_dir = PPL_DIR / f"wikitext2_{mod}" / "gpt2"
        print(f"\n[{idx}/{len(pending)}] Starting PPL evaluation: {mod}...", flush=True)
        
        cmd = [
            python_bin,
            str(eval_script),
            "--model", "models/gpt2",
            "--module", mod,
            "--output_dir", str(out_dir),
        ]
        
        try:
            res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if res.returncode == 0:
                print(f"[PPL Queue] Successfully finished {mod}.", flush=True)
            else:
                print(f"[PPL Queue] Non-zero return code ({res.returncode}) for {mod}.", flush=True)
        except Exception as e:
            print(f"[PPL Queue] Error running {mod}: {e}", flush=True)

if __name__ == "__main__":
    main()
