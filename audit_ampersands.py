import glob
import re

bad_count = 0
for f in glob.glob('thesis_final/**/*.tex', recursive=True):
    with open(f, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    in_tab = False
    for idx, l in enumerate(lines):
        if any(k in l for k in [r'\begin{tabular}', r'\begin{array}', r'\begin{matrix}', r'\begin{align}', r'\begin{align*}', r'\begin{alignat}']):
            in_tab = True
        if any(k in l for k in [r'\end{tabular}', r'\end{array}', r'\end{matrix}', r'\end{align}', r'\end{align*}', r'\end{alignat}']):
            in_tab = False
            continue
        if not in_tab:
            # Check for unescaped &
            # Ignore comments
            clean_l = re.sub(r'(?<!\\)%.*', '', l)
            if re.search(r'(?<!\\)&', clean_l):
                print(f'{f}:{idx+1}: {l.strip()}')
                bad_count += 1

print(f'Total unescaped ampersands: {bad_count}')
