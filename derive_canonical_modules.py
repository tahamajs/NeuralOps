import sys
import importlib
from pathlib import Path

proj_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(proj_dir))
sys.path.insert(0, str(proj_dir / "Codes"))

# Import registry
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

attentions_dir = proj_dir / "Codes" / "framework" / "attentions"
compression_dir = proj_dir / "Codes" / "framework" / "compression_engine"

for f in attentions_dir.glob("*.py"):
    base = f.stem
    if not base.startswith("__") and not base.startswith("._"):
        try:
            importlib.import_module(f"framework.attentions.{base}")
        except Exception as e:
            print(f"Warning importing attention module {base}: {e}")

for f in compression_dir.glob("*.py"):
    base = f.stem
    if not base.startswith("__") and not base.startswith("._"):
        try:
            importlib.import_module(f"framework.compression_engine.{base}")
        except Exception as e:
            print(f"Warning importing compressor module {base}: {e}")

print("=== RAW REGISTRY ENTRIES ===")
print("ATTENTION_REGISTRY count:", len(ATTENTION_REGISTRY))
print("COMPRESSOR_REGISTRY count:", len(COMPRESSOR_REGISTRY))

# Deduplicate by literal object identity (Python id of class/factory)
id_to_object = {}
id_to_names = {}

# Merge entries
all_entries = list(ATTENTION_REGISTRY.items()) + list(COMPRESSOR_REGISTRY.items())

for name, obj in all_entries:
    # Filter dummy placeholder registration if any
    if name in ["your_attention_name", "your_compressor_name"]:
        continue
    obj_id = id(obj)
    id_to_object[obj_id] = obj
    if obj_id not in id_to_names:
        id_to_names[obj_id] = []
    id_to_names[obj_id].append(name)

canonical_list = ["full"]
alias_map = {}

# Priority helper suffixes to de-prioritize when selecting canonical name
helper_suffixes = ('_inject', '_compressor', '_attention', '_attn', '_module', '_fn', '_quantizer', '_pruner')

for obj_id, names in id_to_names.items():
    # Sort names to pick cleanest name
    def sort_key(n):
        has_suffix = any(n.endswith(suf) for suf in helper_suffixes)
        return (has_suffix, len(n), n)

    sorted_names = sorted(names, key=sort_key)
    canonical_name = sorted_names[0]
    if canonical_name != "full":
        canonical_list.append(canonical_name)
    alias_map[canonical_name] = sorted_names

canonical_list = sorted(list(set(canonical_list)))

print(f"\n=== CANONICAL MODULE DERIVATION (OBJECT IDENTITY) ===")
print(f"Total Unique Injected Modules (+ 'full'): {len(canonical_list)}")

# Show aliases for some modules to verify object-identity grouping
print("\nSample Object Identity Groups (Canonical -> Aliases sharing identical object):")
sample_checks = ["adaptive_span", "svd_linear_attention", "aegis_v6", "res_svd", "cosformer", "flash3", "eat"]
for sc in sample_checks:
    found = False
    for can, aliases in alias_map.items():
        if sc in aliases:
            print(f"  {can} (id={id(ATTENTION_REGISTRY.get(sc) or COMPRESSOR_REGISTRY.get(sc))}): {aliases}")
            found = True
            break

# Write to canonical_modules.txt
with open("canonical_modules.txt", "w") as f:
    for mod in canonical_list:
        f.write(f"{mod}\n")

print(f"\nSuccessfully wrote {len(canonical_list)} canonical modules to canonical_modules.txt")
