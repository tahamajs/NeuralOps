import inspect, os, glob, sys

REPO_DIR = Path("D:/SVD_Project") if sys.platform == "win32" else Path.cwd()
sys.path.insert(0, str(REPO_DIR / "Codes"))

import framework.attentions
import framework.compression_engine
import pkgutil

# Pre-import all modules
for _, name, _ in pkgutil.iter_modules(framework.attentions.__path__):
    if not name.startswith("_"):
        try:
            __import__(f"framework.attentions.{name}")
        except Exception:
            pass

for _, name, _ in pkgutil.iter_modules(framework.compression_engine.__path__):
    if not name.startswith("_"):
        try:
            __import__(f"framework.compression_engine.{name}")
        except Exception:
            pass

from framework.core.registry import COMPRESSOR_REGISTRY, ATTENTION_REGISTRY

def has_rank_arg(cls):
    try:
        sig = inspect.signature(cls.__init__)
        return any(p in sig.parameters for p in ("rank", "svd_rank", "cur_rank", "rank_qkv", "r", "lora_rank", "lora_r"))
    except Exception:
        return False

def has_completed_run(name):
    patterns = [
        f"results/unified_*/*/{name}/summary.csv",
        f"results/unified_*/*/checkpoints/{name}/weights.pt",
        f"results/rank_ablation/{name}/**/summary.csv",
        f"results/*_agnews_baseline/{name}/summary.csv",
        f"results/distilbert_agnews_baseline/{name}/summary.csv",
        f"results/unified_{name}/**/summary.csv",
    ]
    for p in patterns:
        if glob.glob(str(REPO_DIR / p), recursive=True):
            return True
    return False

candidates = set()
for reg in (COMPRESSOR_REGISTRY, ATTENTION_REGISTRY):
    for name, cls in getattr(reg, "_registry", {}).items():
        base = name.replace("_inject", "").replace("_factory", "").replace("_compressor", "").replace("_attention", "")
        try:
            if has_rank_arg(cls) and has_completed_run(base):
                candidates.add(base)
        except Exception:
            pass

# Guarantee top rank-ablation thesis modules if present
priority = ["svd_linear", "svd_llm_v2", "asvd", "aasvd", "a3", "mla", "low_rank", "sdrm", "compressed_mha", "fused_svd_kernel", "kronecker", "hamming", "galerkin", "nasc", "svd_mamba", "svd_compressed", "dobi", "optimal_subspace_projection"]
for p in priority:
    if has_completed_run(p):
        candidates.add(p)

out_file = REPO_DIR / "papers/_rank_ablation/modules.txt"
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    for m in sorted(candidates)[:20]:
        f.write(m + "\n")

print(f"Wrote {len(sorted(candidates)[:20])} modules to {out_file}")
for m in sorted(candidates)[:20]:
    print(f"  - {m}")
