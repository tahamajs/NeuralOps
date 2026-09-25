import sys
sys.path.insert(0, 'Codes')

from framework.core.registry import COMPRESSOR_REGISTRY
import framework.compression_engine

OTHER_MODULES = [
    "aasvd",
    "uniql_compressor",
    "compot_compressor",
    "grecodec_compressor",
    "kronecker_compressor",
    "mgie_svd_compressor",
    "saes_svd_v2",
    "rpca_compressor",
    "direct_svd",
    "cur_compressor",
    "erc_svd",
    "frandi_compressor",
    "salr",
    "oats_compressor",
    "mlorq_compressor",
    "modegpt_compressor",
    "hybrid_svd_layer",
    "adaptive_ranks",
    "admm_compressor",
    "come",
    "cr_compressor",
    "cure",
    "rotational_slicegpt_int4",
    "subspace_projection",
]

print(f"{'Module':<28} | {'In Registry':<12} | {'Callable/Class'}")
print("-" * 60)
valid_modules = []
for m in OTHER_MODULES:
    in_reg = m in COMPRESSOR_REGISTRY
    obj = COMPRESSOR_REGISTRY.get(m) if in_reg else None
    obj_str = str(type(obj)) if obj else "None"
    print(f"{m:<28} | {str(in_reg):<12} | {obj_str}")
    if in_reg and obj is not None:
        valid_modules.append(m)

print(f"\nTotal valid modules ready for sweep: {len(valid_modules)} / {len(OTHER_MODULES)}")
with open("audit/cat5a_batch2_modules.txt", "w", encoding="utf-8") as f:
    for m in valid_modules:
        f.write(m + "\n")
print("Saved to audit/cat5a_batch2_modules.txt")
