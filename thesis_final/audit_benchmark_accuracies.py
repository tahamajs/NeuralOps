import os, glob, re
import pandas as pd

with open('chapter_attentions.tex') as f:
    lines = [l.strip() for l in f if l.strip().startswith(r'\input{')]

print(f'Auditing {len(lines)} active modules in chapter_attentions.tex...')

# Also load top_100_results.csv to cross reference real benchmark accuracy
top_csv_path = '/Users/tahamajs/Documents/uni/Project/top_100_results.csv'
top_results = {}
if os.path.exists(top_csv_path):
    df = pd.read_csv(top_csv_path, header=None)
    for idx, row in df.iterrows():
        mod_name = str(row[0]).strip()
        status = str(row[1]).strip()
        acc = row[2]
        try:
            acc_val = float(acc)
        except:
            acc_val = None
        top_results[mod_name] = {'status': status, 'accuracy': acc_val}

print(f'Loaded {len(top_results)} results from top_100_results.csv')

bad_modules = []
good_modules = []

for l in lines:
    fpath = l.split('{')[1].split('}')[0]
    mod = os.path.basename(fpath).replace('.tex', '')
    stem = mod.replace('_attention', '')

    with open(fpath, 'r', encoding='utf-8') as f:
        c = f.read()

    # Find table
    start = c.find(r'\begin{tabular}')
    end = c.find(r'\end{tabular}')
    table_str = c[start:end] if start != -1 and end != -1 else ''

    # Check for FAIL, 0.0, 0%, 0.00%
    is_bad = False
    reasons = []

    if 'FAIL' in table_str or 'ناموفق' in table_str or 'شکست' in table_str:
        is_bad = True
        reasons.append('Explicit FAIL in table')

    # Look for 0.00% or 0% or 0.0%
    for match in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:%|٪)', table_str):
        val = float(match.group(1))
        if val == 0.0:
            is_bad = True
            reasons.append('0% accuracy in table')
            break

    # Look for 0.00 in table cells
    for cell in table_str.split('&'):
        cell_clean = cell.strip()
        if cell_clean in ['0', '0.0', '0.00', '0.000', '0.0000']:
            is_bad = True
            reasons.append(f'Zero cell: {cell_clean}')
            break

    # Cross-reference with top_100_results.csv if available
    for key in [mod, stem, mod.replace('_module', ''), mod.replace('_attn', '')]:
        if key in top_results:
            tr = top_results[key]
            if tr['status'] == 'FAIL' or (tr['accuracy'] is not None and tr['accuracy'] == 0.0):
                is_bad = True
                reasons.append(f"top_100_results.csv reports status={tr['status']}, acc={tr['accuracy']}")
            break

    if is_bad:
        bad_modules.append((mod, list(set(reasons))))
    else:
        good_modules.append(mod)

print(f'\nTotal GOOD modules: {len(good_modules)}')
print(f'Total BAD / ZERO / FAILED modules: {len(bad_modules)}')

print('\nBAD / ZERO / FAILED modules:')
for m, reasons in bad_modules:
    print(f'  {m}: {reasons}')
