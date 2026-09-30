import sys
import os
from pathlib import Path
import copy
import time
import torch

sys.path.insert(0, str(Path("Codes").resolve()))
from transformers import AutoModelForCausalLM, AutoTokenizer
from framework.eval_all_models_datasets import apply_module_to_model
from framework.run_gpt2_sweep import get_all_decoder_modules, load_deprecated_modules

def main():
    deprecated = load_deprecated_modules()
    all_mods = [m for m in get_all_decoder_modules("attentions") if m != "full" and m not in deprecated]
    print(f"Testing {len(all_mods)} active attention modules on GPT-2 on CUDA with uniform fp16...", flush=True)

    tok = AutoTokenizer.from_pretrained("models/gpt2_wikitext")
    base = AutoModelForCausalLM.from_pretrained("models/gpt2_wikitext").half().cuda()
    input_ids = torch.randint(0, 1000, (1, 64), device="cuda")

    results = {"pass": [], "fail": [], "nan": []}

    for i, mod in enumerate(all_mods, 1):
        if "moe" in mod:
            continue
        try:
            m = copy.deepcopy(base)
            m, stats = apply_module_to_model(m, mod, "attention", rank=64, device="cuda")
            m.to(device="cuda", dtype=torch.float16)
            m.eval()
            with torch.no_grad():
                out = m(input_ids)
                logits = out.logits
                if not torch.isfinite(logits).all():
                    results["nan"].append(mod)
                    print(f"[{i}/{len(all_mods)}] {mod}: NAN/INF", flush=True)
                else:
                    results["pass"].append(mod)
                    print(f"[{i}/{len(all_mods)}] {mod}: PASS", flush=True)
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            results["fail"].append((mod, str(e)[:150]))
            print(f"[{i}/{len(all_mods)}] {mod}: FAIL -> {str(e)[:100]}", flush=True)

    print("\n" + "="*50, flush=True)
    print(f"PASS: {len(results['pass'])}", flush=True)
    print(f"NAN/INF: {len(results['nan'])}", flush=True)
    print(f"FAIL: {len(results['fail'])}", flush=True)
    for mod, err in results["fail"]:
        print(f"  {mod}: {err}", flush=True)
    if results["nan"]:
        print("NAN modules:", results["nan"], flush=True)

if __name__ == "__main__":
    main()
