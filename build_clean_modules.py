import os
from pathlib import Path

# Scan results directory for unified_* and agnews_unified_*
repo_dir = Path("D:/SVD_Project") if os.path.exists("D:/SVD_Project") else Path(".")
res_dir = repo_dir / "results"

candidates = set()
for item in res_dir.iterdir():
    if not item.is_dir():
        continue
    name = item.name
    mod = None
    if name.startswith("agnews_unified_"):
        mod = name[len("agnews_unified_"):]
    elif name.startswith("unified_"):
        mod = name[len("unified_"):]
    
    if mod:
        # Check if it has completed weights or summary
        has_run = any(item.glob("**/summary.csv")) or any(item.glob("**/weights.pt"))
        if has_run:
            candidates.add(mod)

# Hard filter / priority for known rank-parameterized modules
rank_priority = [
    "svd_linear", "svd_llm_v2", "asvd", "aasvd", "a3", "mla", 
    "low_rank", "sdrm", "compressed_mha", "kronecker", "hamming", 
    "galerkin", "nasc", "svd_mamba", "svd_compressed", "dobi", 
    "optimal_subspace_projection", "adaptive_svd_attention", "primal", "eat"
]

selected = []
for p in rank_priority:
    if p in candidates:
        selected.append(p)

for c in sorted(candidates):
    if c not in selected and len(selected) < 20:
        selected.append(c)

selected = selected[:20]

out_dir = repo_dir / "papers" / "_rank_ablation"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "modules.txt"

with open(out_file, "w", encoding="utf-8") as f:
    for s in selected:
        f.write(s + "\n")

print(f"Successfully wrote {len(selected)} modules to {out_file}:")
for s in selected:
    print(f"  - {s}")
