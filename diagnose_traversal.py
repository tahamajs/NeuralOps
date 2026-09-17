import glob, re, os

files = sorted(glob.glob('Codes/framework/attentions/*.py'))
injectors = []
has_traversal = []
missing = []

for f in files:
    try:
        src = open(f, encoding='utf-8', errors='ignore').read()
    except Exception:
        continue
    if re.search(r'def inject_\w+\s*\(', src):
        injectors.append(f)
        has_it = bool(re.search(
            r'target\s*=\s*layer if layer is not None else model|'
            r'layer_list\s*=\s*None|'
            r'for idx, lyr in enumerate\(layer_list\)|'
            r'for lyr in layer_list',
            src
        ))
        if has_it:
            has_traversal.append(f)
        else:
            missing.append(f)

print(f"TOTAL_INJECTORS: {len(injectors)}")
print(f"WITH_TRAVERSAL: {len(has_traversal)}")
for f in has_traversal:
    print(f"  OK {f}")
print(f"WITHOUT_TRAVERSAL: {len(missing)}")
for f in missing:
    print(f"  BUG {f}")
