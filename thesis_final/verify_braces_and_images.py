import os, glob, re

files = glob.glob('thesis_final/attentions_latex/individual_modules/*.tex') + ['thesis_final/master_thesis.tex']

total_errors = 0
for f in files:
    with open(f, 'r', encoding='utf-8') as fl:
        c = fl.read()
    
    # check brace balance
    open_c = c.count('{')
    close_c = c.count('}')
    if open_c != close_c:
        print(f'BRACE MISMATCH in {f}: {open_c} vs {close_c}')
        total_errors += 1
    
    # check includegraphics path exists
    imgs = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', c)
    for img in imgs:
        p = os.path.join('thesis_final', img)
        if not os.path.exists(p):
            print(f'MISSING IMAGE in {f}: {img} (checked {p})')
            total_errors += 1

print(f'Verification finished. Total errors: {total_errors}')
