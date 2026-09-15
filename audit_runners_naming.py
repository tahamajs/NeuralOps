import os
import glob
import re

runners = sorted(glob.glob("/Users/tahamajs/Documents/uni/Project/Codes/runners/run_all_4_*.py"))
mismatches = []

for r in runners:
    base = os.path.basename(r)[len("run_all_4_"):-3]
    with open(r, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Check out_dir patterns
    out_dirs = re.findall(r'out_dir["\']?\s*:\s*r?["\']([^"\']+)["\']', content)
    for od in out_dirs:
        # Expected unified_<base>\...
        parts = od.replace('/', '\\').split('\\')
        if len(parts) >= 2 and parts[0].startswith("results"):
            target_unified = parts[1]
        elif len(parts) >= 1 and parts[0].startswith("unified_"):
            target_unified = parts[0]
        else:
            continue
            
        expected = f"unified_{base}"
        if target_unified != expected and not target_unified.endswith("_rerun"):
            mismatches.append((r, base, target_unified))
            break

print(f"Total directory mismatches in runners: {len(mismatches)}")
for r, b, t in mismatches:
    print(f"- {os.path.basename(r)}: expected unified_{b}, but writes to {t}")

