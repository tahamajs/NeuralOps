import os, glob, re
import pandas as pd

all_tex = glob.glob('attentions_latex/individual_modules/*.tex')
print(f'Total tex files in individual_modules: {len(all_tex)}')

results = []

for fpath in all_tex:
    mod = os.path.basename(fpath).replace('.tex', '')
    stem = mod.replace('_attention', '')
    with open(fpath, 'r', encoding='utf-8') as f:
        c = f.read()

    start = c.find(r'\begin{tabular}')
    end = c.find(r'\end{tabular}')
    table_str = c[start:end] if start != -1 and end != -1 else ''

    # Check for FAIL
    has_fail = 'FAIL' in table_str or 'ناموفق' in table_str or 'شکست' in table_str

    # Extract all percentages like 94.23\% or 0.00\% or 85.5% or 88٪
    # Matches digits followed by optional .digits then optional \ then % or ٪
    acc_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:\\%|%|٪)', table_str)
    acc_values = [float(x) for x in acc_matches]

    has_zero = any(a == 0.0 for a in acc_values)
    is_all_zero = len(acc_values) > 0 and all(a == 0.0 for a in acc_values)
    min_acc = min(acc_values) if acc_values else 0.0
    avg_acc = sum(acc_values) / len(acc_values) if acc_values else 0.0

    # Also check if F1 is reported and status is PASS
    is_pure_pass = (not has_fail) and (not has_zero) and len(acc_values) > 0 and min_acc > 50.0

    results.append({
        'mod': mod,
        'fpath': fpath,
        'has_fail': has_fail,
        'has_zero': has_zero,
        'is_all_zero': is_all_zero,
        'acc_values': acc_values,
        'min_acc': min_acc,
        'avg_acc': avg_acc,
        'is_pure_pass': is_pure_pass,
        'table_str': table_str
    })

print(f'\nTotal modules: {len(results)}')
zero_or_fail = [r for r in results if r['has_fail'] or r['has_zero']]
pure_passes = [r for r in results if r['is_pure_pass']]

print(f'Modules with ANY zero or FAIL: {len(zero_or_fail)}')
print(f'Modules that are 100% PURE PASS (min accuracy > 50%, no fails, no zeros): {len(pure_passes)}')

# Check current active modules in chapter_attentions.tex
with open('chapter_attentions.tex') as f:
    active_mods = [os.path.basename(l.split('{')[1].split('}')[0]).replace('.tex', '') for l in f if 'input{' in l]

print(f'\nActive modules count in chapter_attentions.tex: {len(active_mods)}')
active_zero_or_fail = [r for r in results if r['mod'] in active_mods and (r['has_fail'] or r['has_zero'])]
active_pure_pass = [r for r in results if r['mod'] in active_mods and r['is_pure_pass']]

print(f'Active modules with ZERO or FAIL: {len(active_zero_or_fail)}')
print(f'Active modules that are PURE PASS: {len(active_pure_pass)}')

print('\nList of ACTIVE modules that have ZERO or FAIL:')
for r in active_zero_or_fail:
    print(f'  {r["mod"]:30s} | fail={r["has_fail"]} | has_zero={r["has_zero"]} | accs={r["acc_values"]}')

print('\nOther available PURE PASS modules that can replace them:')
avail_pure = [r for r in pure_passes if r['mod'] not in active_mods]
for r in avail_pure[:20]:
    print(f'  {r["mod"]:30s} | avg={r["avg_acc"]:5.2f}% | min={r["min_acc"]:5.2f}% | accs={r["acc_values"]}')
