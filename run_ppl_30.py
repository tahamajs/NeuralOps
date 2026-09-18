import subprocess
import sys
import json
from pathlib import Path

modules = [
    "lora_attention", "lsh_attention", "compressed_mha", "gqa", "aegis_hybrid",
    "full_attention", "svd_linear_attention", "reap_moe", "minimal_prism", "halo_regla",
    "cosformer", "performer_attention", "samba_m3", "mamba", "hla_inject",
    "gated_deltanet_block", "galerkin", "phantom", "jamba", "unified_rope",
    "adaptive_svd_attention", "moba", "mla_attention", "vig_attention", "zeros_attention",
    "aasvd", "slicegpt", "sparsegpt", "proxsparse_pruner", "saes_svd"
]

print(f"Starting PPL sweep for {len(modules)} modules...", flush=True)

for idx, mod in enumerate(modules, 1):
    out_dir = f"results/wikitext2_{mod}/gpt2/"
    cmd = [
        "python", "-u", "Codes/framework/eval_perplexity.py",
        "--model", "models/gpt2",
        "--module", mod,
        "--output_dir", out_dir
    ]
    print(f"[{idx}/{len(modules)}] Evaluating {mod}...", flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"FAILED: {mod} exited with code {res.returncode}", flush=True)
        # Record failure json if not created
        p = Path(out_dir) / "ppl_result.json"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump({"module": mod, "perplexity": None, "error": f"Exit code {res.returncode}"}, f, indent=2)

print("PPL sweep finished successfully!", flush=True)
