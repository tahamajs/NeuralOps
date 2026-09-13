import os
import re

MOD_DIR = '/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules'
with open('/Users/tahamajs/Documents/uni/Project/thesis_final/chapter_attentions.tex', 'r', encoding='utf-8') as f:
    inputs = [l.strip() for l in f if l.strip().startswith(r'\input{')]
modules = [l.split('{')[1].split('}')[0].split('/')[-1].replace('.tex', '') for l in inputs]

errors = []
for m in modules:
    p = os.path.join(MOD_DIR, f'{m}.tex')
    with open(p, 'r', encoding='utf-8') as f:
        txt = f.read()
    
    open_b = txt.count('{')
    close_b = txt.count('}')
    if open_b != close_b:
        errors.append(f'{m}: Brace mismatch ({open_b} open vs {close_b} close)')
        
    begins = re.findall(r'\\begin\{([a-zA-Z0-9_\*]+)\}', txt)
    ends = re.findall(r'\\end\{([a-zA-Z0-9_\*]+)\}', txt)
    if sorted(begins) != sorted(ends):
        errors.append(f'{m}: Environment mismatch: begins={sorted(begins)} ends={sorted(ends)}')

print(f'Total syntax errors found: {len(errors)}')
for e in errors:
    print('  ', e)
