import os, re

with open('thesis_final/chapter_attentions.tex') as f:
    modules = [l.strip().replace(r'\input{', '').rstrip('}') for l in f if 'input{' in l]

errors = 0
for m in modules:
    if not os.path.exists(m):
        print('NOT FOUND:', m)
        errors += 1
        continue
    with open(m, 'r', encoding='utf-8') as fl:
        c = fl.read()
    o = c.count('{')
    cl = c.count('}')
    if o != cl:
        print(f'BRACE MISMATCH in {os.path.basename(m)}: {o} vs {cl}')
        errors += 1
    
    # check images
    imgs = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', c)
    for img in imgs:
        p = os.path.join('thesis_final', img)
        if not os.path.exists(p):
            print(f'MISSING IMAGE in {os.path.basename(m)}: {img}')
            errors += 1

print(f'Checked {len(modules)} modules. Total errors: {errors}')
