import os
import ast
import glob
import importlib
import traceback

print("=== VERIFICATION REPORT ===")

def verify_ast():
    print("\n--- 1. AST Parsing ---")
    files = glob.glob('Codes/framework/**/*.py', recursive=True)
    errors = 0
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                ast.parse(file.read())
        except Exception as e:
            print(f"AST Error in {f}: {e}")
            errors += 1
    if errors == 0:
        print("OK: All files passed AST parsing successfully.")

def verify_strings():
    print("\n--- 2. Fix Verification ---")
    checks = {
        'Codes/framework/attentions/fused_svd_kernel.py': ['fill_diagonal_'],
        'Codes/framework/attentions/gated_delta_net.py': ['chunk_size = 32'],
        'Codes/framework/attentions/rala_attention.py': ['self.alpha.requires_grad = False'],
        'Codes/framework/attentions/compressed_attention.py': ['is_nested_encoder'],
        'Codes/framework/attentions/svd_linear_attention.py': ['192'],
        'Codes/framework/compression_engine/cur_moe_compressor.py': ['visited'],
        'Codes/framework/attentions/flash_attention3_fp8.py': ['target_mod.freeze_base_weights']
    }
    for filepath, strings in checks.items():
        if not os.path.exists(filepath):
            print(f"MISSING FILE: {filepath}")
            continue
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        for s in strings:
            if s in content:
                print(f"OK: {filepath} contains '{s}'")
            else:
                print(f"MISSING STRING: {filepath} missing '{s}'")
                
        if filepath == 'Codes/framework/attentions/flash_attention3_fp8.py':
            count = content.count('target_mod.freeze_base_weights()')
            if count > 1:
                print(f"FAIL: {filepath} has {count} freeze_base_weights() calls (should be <= 1)")
            else:
                print(f"OK: {filepath} freeze_base_weights() count is {count}")
                
def verify_imports():
    print("\n--- 3. Import Check ---")
    import sys
    sys.path.insert(0, os.getcwd())
    try:
        from Codes.framework.attentions import __all__ as att_all
        print("OK: Successfully imported Codes.framework.attentions")
    except Exception as e:
        print(f"FAIL: Failed to import attentions: {e}")
    try:
        from Codes.framework.compression_engine import __all__ as ce_all
        print("OK: Successfully imported Codes.framework.compression_engine")
    except Exception as e:
        print(f"FAIL: Failed to import compression_engine: {e}")

def verify_results():
    print("\n--- 4. Results Directory Integrity ---")
    res_dir = 'results'
    if not os.path.exists(res_dir):
        print("FAIL: 'results' directory is missing!")
        return
        
    unified = glob.glob(f'{res_dir}/unified_*')
    weights = glob.glob(f'{res_dir}/**/weights.pt', recursive=True)
    summaries = glob.glob(f'{res_dir}/**/summary.csv', recursive=True)
    json_metrics = glob.glob(f'{res_dir}/**/*.json', recursive=True)
    
    print(f"OK: Found {len(unified)} unified log/cache files.")
    print(f"OK: Found {len(weights)} weights.pt checkpoints.")
    print(f"OK: Found {len(summaries)} summary.csv files.")
    print(f"OK: Found {len(json_metrics)} JSON metric files.")

if __name__ == '__main__':
    verify_ast()
    verify_strings()
    verify_imports()
    verify_results()
