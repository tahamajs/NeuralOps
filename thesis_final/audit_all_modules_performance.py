import os, glob, re

all_tex = glob.glob('attentions_latex/individual_modules/*.tex')
print(f'Total tex files in individual_modules: {len(all_tex)}')

results = []

for fpath in all_tex:
    mod = os.path.basename(fpath).replace('.tex', '')
    with open(fpath, 'r', encoding='utf-8') as f:
        c = f.read()

    start = c.find(r'\begin{tabular}')
    end = c.find(r'\end{tabular}')
    table_str = c[start:end] if start != -1 and end != -1 else ''

    # Check for FAIL or 0.00%
    has_any_fail = 'FAIL' in table_str or 'ناموفق' in table_str or 'شکست' in table_str

    # Extract all percentages
    acc_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|٪)', table_str)
    acc_values = [float(x) for x in acc_matches]

    has_zero = any(a == 0.0 for a in acc_values)
    min_acc = min(acc_values) if acc_values else 0.0
    avg_acc = sum(acc_values) / len(acc_values) if acc_values else 0.0

    # Also check if all rows are zero
    is_all_zero = len(acc_values) > 0 and all(a == 0.0 for a in acc_values)

    results.append({
        'mod': mod,
        'fpath': fpath,
        'has_any_fail': has_any_fail,
        'has_zero': has_zero,
        'is_all_zero': is_all_zero,
        'acc_values': acc_values,
        'min_acc': min_acc,
        'avg_acc': avg_acc,
    })

all_zero_mods = [r for r in results if r['is_all_zero']]
any_zero_or_fail = [r for r in results if r['has_zero'] or r['has_any_fail']]
strictly_positive = [r for r in results if not r['has_zero'] and not r['has_any_fail'] and len(r['acc_values']) > 0]

print(f'Total modules analyzed: {len(results)}')
print(f'Modules with ALL ZERO accuracy: {len(all_zero_mods)}')
print(f'Modules with ANY zero or FAIL: {len(any_zero_or_fail)}')
print(f'Modules with STRICTLY POSITIVE (>0), 100% PASS accuracy: {len(strictly_positive)}')

# Check which of the current 100 active modules in chapter_attentions.tex fall into these
with open('chapter_attentions.tex') as f:
    active_mods = [os.path.basename(l.split('{')[1].split('}')[0]).replace('.tex', '') for l in f if 'input{' in l]

active_results = [r for r in results if r['mod'] in active_mods]
active_bad = [r for r in active_results if r['has_zero'] or r['has_any_fail']]
active_good = [r for r in active_results if not r['has_zero'] and not r['has_any_fail'] and len(r['acc_values']) > 0]

print(f'\nIn CURRENT chapter_attentions.tex (100 modules):')
print(f'  Good (No zeros, no fail): {len(active_good)}')
print(f'  Bad (Contains zero or fail): {len(active_bad)}')
print('\nBad active modules that MUST be removed or replaced:')
for r in active_bad:
    print(f'  {r["mod"]:30s} | fail={r["has_any_fail"]} | has_zero={r["has_zero"]} | accs={r["acc_values"]}')
