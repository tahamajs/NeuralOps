"""List TOP-LEVEL unique modules only (aggressive dedup)."""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

for p in sorted((HERE / "neuralops" / "attentions").glob("*.py")):
    if p.stem.startswith("_"):
        continue
    try:
        importlib.import_module(f"neuralops.attentions.{p.stem}")
    except Exception:
        pass

for p in sorted((HERE / "neuralops" / "compression_engine").glob("*.py")):
    if p.stem.startswith("_"):
        continue
    try:
        importlib.import_module(f"neuralops.compression_engine.{p.stem}")
    except Exception:
        pass

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

all_keys = set(ATTENTION_REGISTRY.keys()) | set(COMPRESSOR_REGISTRY.keys())

# Exclusion list: aliases, helpers, tests
EXCLUDE_KEYWORDS = [
    "factory", "inject", "_class", "_module", "_helper",
    "test", "debug", "example", "custom", "template",
    "_v2", "_v3", "_v4", "_v5", "_v6",
    "full", "dense",
    "weighted_ensemble", "smart_weighted",
    "your_custom",
]

unique = []
seen_bases = set()

for key in sorted(all_keys):
    kl = key.lower()
    if any(kw in kl for kw in EXCLUDE_KEYWORDS):
        continue
    # Strip common suffixes for dedup
    base = key
    for suf in ["_compressor", "_attention", "_inject", "_layer", "_block", "_outer"]:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    if base in seen_bases:
        continue
    seen_bases.add(base)
    unique.append(key)

unique = sorted(unique)

print(f"=== TOP-LEVEL UNIQUE ({len(unique)}) ===")
for m in unique:
    print(m)
print(f"\n=== TOTAL: {len(unique)} ===")

with open("core_modules.txt", "w") as f:
    for m in unique:
        f.write(m + "\n")
print("Saved to core_modules.txt")
