import os
import glob
import hashlib
from generate_individual_bespoke_module import process_single_module_bespoke, ATTENTIONS_DIR, COMPRESSIONS_DIR

def run_all_sequentially():
    att_files = sorted(glob.glob(os.path.join(ATTENTIONS_DIR, "*.py")))
    comp_files = sorted(glob.glob(os.path.join(COMPRESSIONS_DIR, "*.py")))
    
    all_modules = []
    for f in att_files:
        m = os.path.splitext(os.path.basename(f))[0]
        if not m.startswith("__"):
            all_modules.append((m, "attention"))
            
    for f in comp_files:
        m = os.path.splitext(os.path.basename(f))[0]
        if not m.startswith("__"):
            all_modules.append((m, "compression"))
            
    total = len(all_modules)
    print(f"Starting individual, sequential generation for all {total} modules...")
    
    completed = 0
    hashes = {}
    
    for idx, (m_name, category) in enumerate(all_modules, 1):
        success = process_single_module_bespoke(m_name, category)
        if success:
            completed += 1
            
        if idx % 10 == 0 or idx == total:
            print(f"[{idx}/{total}] Completed individual generation: {completed} successful.")
            
    print(f"\nFinished! All {completed}/{total} modules generated individually and sequentially with verified AST grounding.")

if __name__ == "__main__":
    run_all_sequentially()
