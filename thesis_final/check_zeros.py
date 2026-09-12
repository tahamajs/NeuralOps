import os, re

with open('chapter_attentions.tex') as f:
    lines = [l.strip() for l in f if l.strip().startswith(r'\input{')]

print(f'Total active modules: {len(lines)}')

zero_modules = []
valid_modules = []

for l in lines:
    fpath = l.split('{')[1].split('}')[0]
    mod = os.path.basename(fpath).replace('.tex', '')
    with open(fpath) as f:
        c = f.read()

    start = c.find(r'\begin{tabular}')
    end = c.find(r'\end{tabular}')
    table_str = c[start:end] if start != -1 and end != -1 else ''

    acc_line = ''
    for line in table_str.splitlines():
        if 'Accuracy' in line or 'دقت' in line:
            acc_line = line
            break

    # Look for 0 or 0.0 or 0% or FAIL in table
    has_zero = False
    if 'FAIL' in table_str or 'ناموفق' in table_str:
        has_zero = True
    elif acc_line:
        # Check if table has 0 or 0.0 or 0%
        # e.g. & 0.0 & or & 0 & or & 0.0٪ & or & 0٪
        parts = [p.strip() for p in acc_line.split('&')]
        for p in parts[1:]:
            p_clean = p.replace(r'\%', '').replace('٪', '').replace(r'\\', '').strip()
            try:
                val = float(p_clean)
                if val == 0.0:
                    has_zero = True
            except:
                if p_clean == '0' or p_clean == '0.0' or p_clean == '0.00':
                    has_zero = True

    if has_zero:
        zero_modules.append((mod, acc_line))
    else:
        valid_modules.append((mod, acc_line))

print(f'Valid modules ({len(valid_modules)}):')
for m, a in valid_modules[:20]:
    print('  ', m, '-->', a.strip())

print(f'\nZero / Failed modules ({len(zero_modules)}):')
for m, a in zero_modules:
    print('  ', m, '-->', a.strip())
