import re

fixes = [
    ("/Users/tahamajs/Documents/uni/Project/Codes/runners/run_all_4_adaptive_sparse_attention.py",
     "unified_adaptive_sparse_attention_attention", "unified_adaptive_sparse_attention"),
    ("/Users/tahamajs/Documents/uni/Project/Codes/runners/run_all_4_dobi.py",
     "unified_dobi_svd", "unified_dobi"),
    ("/Users/tahamajs/Documents/uni/Project/Codes/runners/run_all_4_dynamic_entropy.py",
     "unified_dynamic_entropy_svd", "unified_dynamic_entropy"),
    ("/Users/tahamajs/Documents/uni/Project/Codes/runners/run_all_4_samba_m3.py",
     "unified_sambam3", "unified_samba_m3"),
    ("/Users/tahamajs/Documents/uni/Project/Codes/runners/run_all_4_samba_moe.py",
     "unified_sambamoe", "unified_samba_moe"),
    ("/Users/tahamajs/Documents/uni/Project/Codes/runners/run_all_4_unified_rope.py",
     "unified_unified_rope", "unified_rope"),
]

for path, old_str, new_str in fixes:
    with open(path, 'r', encoding='utf-8') as f:
        c = f.read()
    if old_str in c:
        c = c.replace(old_str, new_str)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(c)
        print(f"Fixed: {path}")
    else:
        print(f"Skipped (not found): {path}")

