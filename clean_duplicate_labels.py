import os, re
from collections import Counter

MODULES_DIR = '/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules'

with open('/Users/tahamajs/Documents/uni/Project/thesis_final/chapter_attentions.tex', 'r', encoding='utf-8') as f:
    mods = [line.split('{')[1].split('}')[0].replace('.tex', '').split('/')[-1] for line in f if '\\input{' in line]

for m in mods:
    p = os.path.join(MODULES_DIR, f'{m}.tex')
    with open(p, 'r', encoding='utf-8') as f:
        c = f.read()

    labels = re.findall(r'\\label\{([^}]+)\}', c)
    counts = Counter(labels)
    dups = [k for k, v in counts.items() if v > 1]
    if dups:
        parts = re.split(r'(\\subsubsection\*\{بخش\s*۱۱:[^\}]+\}|\\subsection\*\{بخش\s*۱۱:[^\}]+\})', c)
        if len(parts) >= 3:
            before_s11 = parts[0]
            s11_and_after = ''.join(parts[1:])
            for d in dups:
                if d.startswith('fig:arch_'):
                    new_label = d.replace('fig:arch_', 'fig:flow_')
                    before_s11 = before_s11.replace(f'\\label{{{d}}}', f'\\label{{{new_label}}}', 1)
                elif d.startswith('fig:comp_'):
                    fig_pattern = r'\\begin\{figure\}[\s\S]*?\\label\{' + re.escape(d) + r'\}[\s\S]*?\\end\{figure\}'
                    before_s11 = re.sub(fig_pattern, '', before_s11)
            
            new_c = before_s11 + s11_and_after
            with open(p, 'w', encoding='utf-8') as f:
                f.write(new_c)
            print(f'Cleaned duplicates in {m}')

print('Checking duplicate labels again:')
has_any = False
for m in mods:
    p = os.path.join(MODULES_DIR, f'{m}.tex')
    with open(p, 'r', encoding='utf-8') as f:
        c = f.read()
    labels = re.findall(r'\\label\{([^}]+)\}', c)
    counts = Counter(labels)
    dups = [k for k, v in counts.items() if v > 1]
    if dups:
        print(f'  {m}: STILL HAS DUPLICATES: {dups}')
        has_any = True
if not has_any:
    print('ALL MODULES HAVE ZERO DUPLICATE LABELS!')
