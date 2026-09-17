import sys
from pathlib import Path
import csv
import io
import inspect

cwd = Path.cwd()
if str(cwd / "Codes") not in sys.path:
    sys.path.insert(0, str(cwd / "Codes"))

# Trigger the auto-imports
import framework.attentions
import framework.compression_engine
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

def check_rank_args():
    results = []
    
    target_modules = ["asvd", "aasvd", "mla", "compressed_mha", "galerkin", "hamming", "nasc", "dobi", "kronsvd", "kronecker"]
    
    for mod_name in target_modules:
        target_class = None
        if mod_name in ATTENTION_REGISTRY:
            target_class = ATTENTION_REGISTRY.get(mod_name)
        elif mod_name in COMPRESSOR_REGISTRY:
            target_class = COMPRESSOR_REGISTRY.get(mod_name)
            
        if target_class is None:
            results.append({"module": mod_name, "found": False, "has_rank_arg": False, "arg_name": ""})
            continue
            
        try:
            sig = inspect.signature(target_class.__init__)
            params = sig.parameters
            rank_args = [p for p in params if "rank" in p.lower() or p in ["r", "k", "n_components"]]
            results.append({
                "module": mod_name,
                "found": True,
                "has_rank_arg": len(rank_args) > 0,
                "arg_name": ", ".join(rank_args) if rank_args else "NONE"
            })
        except Exception as e:
            results.append({"module": mod_name, "found": True, "has_rank_arg": "ERROR", "arg_name": str(e)})
            
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["module", "found", "has_rank_arg", "arg_name"])
    writer.writeheader()
    writer.writerows(results)
    print(output.getvalue())

if __name__ == "__main__":
    check_rank_args()
