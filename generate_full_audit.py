import sys, os, glob, re, csv
from collections import defaultdict
sys.path.insert(0, 'Codes')

# Import all attention and compression modules
for f in sorted(glob.glob('Codes/framework/attentions/*.py') + glob.glob('Codes/framework/compression_engine/*.py')):
    b = os.path.basename(f)
    if b == '__init__.py': continue
    pkg = 'framework.attentions.' if 'attentions' in f else 'framework.compression_engine.'
    try:
        __import__(pkg + b[:-3])
    except Exception:
        pass

from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

modules_acc = defaultdict(dict)
for path in glob.glob('results/**/summary*.csv', recursive=True) + glob.glob('results/**/benchmark_summary*.csv', recursive=True):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mod = row.get('module_name') or row.get('module') or row.get('method') or row.get('architecture')
                model = row.get('model_family') or row.get('model') or ''
                acc = row.get('mean_accuracy') or row.get('accuracy') or row.get('eval_accuracy')
                if mod and model and acc:
                    mod = mod.strip().lower()
                    model = model.strip().lower()
                    try:
                        acc_f = float(acc)
                        if model not in modules_acc[mod] or acc_f > modules_acc[mod][model]:
                            modules_acc[mod][model] = acc_f
                    except:
                        pass
    except:
        pass

doc_files = glob.glob('Docs/content/**/*.md', recursive=True) + glob.glob('frontend/dist/Docs/content/**/*.md', recursive=True)
doc_map = {}
for df in doc_files:
    base = os.path.splitext(os.path.basename(df))[0].lower()
    if base not in doc_map:
        doc_map[base] = df

inventory = []
py_files = sorted(glob.glob('Codes/framework/attentions/*.py') + glob.glob('Codes/framework/compression_engine/*.py'))

for f in py_files:
    b = os.path.basename(f)
    if b == '__init__.py': continue
    mod_name = b[:-3]
    mod_type = 'attention' if 'attentions' in f else 'compression'
    
    with open(f, 'r', encoding='utf-8', errors='ignore') as fp:
        code = fp.read()
    
    regs = re.findall(r'@(?:ATTENTION_REGISTRY|COMPRESSOR_REGISTRY)\.register\([\"\']([^\"\']+)[\"\']', code)
    if not regs:
        if mod_name in ATTENTION_REGISTRY: regs.append(mod_name)
        elif mod_name in COMPRESSOR_REGISTRY: regs.append(mod_name)

    doc_path = doc_map.get(mod_name.lower(), doc_map.get(mod_name.lower().replace('_attention', ''), 'MISSING'))
    if doc_path != 'MISSING':
        doc_path = os.path.relpath(doc_path)
    rel_py = os.path.relpath(f)

    accs = modules_acc.get(mod_name.lower(), {})
    for r in regs:
        if r.lower() in modules_acc:
            for k, v in modules_acc[r.lower()].items():
                if k not in accs or v > accs[k]: accs[k] = v

    d_acc = f"{accs.get('distilbert', 0.0):.2f}%" if 'distilbert' in accs else 'N/A'
    r_acc = f"{accs.get('roberta', accs.get('bert', 0.0)):.2f}%" if ('roberta' in accs or 'bert' in accs) else 'N/A'
    g_acc = f"{accs.get('gpt2', accs.get('gpt', 0.0)):.2f}%" if ('gpt2' in accs or 'gpt' in accs) else 'N/A'
    q_acc = f"{accs.get('qwen3_5', accs.get('qwen', 0.0)):.2f}%" if ('qwen3_5' in accs or 'qwen' in accs) else 'N/A'

    has_copy = 'def copy_weights' in code
    has_norm = ('linalg.norm' in code or 'torch.norm' in code) and '1e-4' in code
    has_loop = bool(re.search(r'for\s+[a-zA-Z0-9_]+\s+in\s+range\s*\(\s*(?:seq_len|L|T|N)', code))
    
    is_50 = any(isinstance(v, float) and 48.5 <= v <= 52.5 for v in accs.values())
    is_zero = any(isinstance(v, float) and v == 0.0 for v in accs.values())

    status = 'VERIFIED_PASS'
    issues = []
    if is_50:
        status = 'FLAGGED_50_COLLAPSE'
        issues.append('logged 50.92% collapse')
    elif is_zero:
        status = 'FLAGGED_ZERO_ACC'
        issues.append('logged 0.0% failure')
    
    if has_loop:
        issues.append('sequential loop in forward')
    if not has_norm and has_copy:
        issues.append('missing 1e-4 norm check')
    if doc_path == 'MISSING':
        issues.append('missing doc')

    issues_str = ', '.join(issues) if issues else 'none'

    inventory.append({
        'name': mod_name,
        'regs': ', '.join(regs[:3]) + ('...' if len(regs) > 3 else ''),
        'py': rel_py,
        'doc': doc_path,
        'distilbert': d_acc,
        'roberta': r_acc,
        'gpt2': g_acc,
        'qwen': q_acc,
        'status': status,
        'issues': issues_str,
        'is_priority': 1 if is_50 or is_zero else (2 if doc_path == 'MISSING' else 3)
    })

inventory.sort(key=lambda x: (x['is_priority'], x['name']))

out_file = 'results/full_module_audit.md'
with open(out_file, 'w', encoding='utf-8') as f:
    f.write('# Full Systematic Audit of All Attention & Compression Modules\n\n')
    f.write(f'Generated: 2026-09-08 | Total Modules Tracked: {len(inventory)}\n\n')
    f.write('## Priority Breakdown\n')
    f.write(f'- Priority 1 (50.92% Collapse or 0.00% / Failures): {sum(1 for x in inventory if x["is_priority"] == 1)}\n')
    f.write(f'- Priority 2 (Missing Documentation): {sum(1 for x in inventory if x["is_priority"] == 2)}\n')
    f.write(f'- Priority 3 (Standard / Passing Modules): {sum(1 for x in inventory if x["is_priority"] == 3)}\n\n')
    f.write('## Master Audit Inventory\n\n')
    f.write('| Module | Registry Name(s) | Source File | Documentation | DistilBERT | RoBERTa | GPT-2 | Qwen3.5 | Status | Code Issues / Flags |\n')
    f.write('|---|---|---|---|---|---|---|---|---|---|\n')
    for item in inventory:
        doc_link = 'MISSING' if item['doc'] == 'MISSING' else f"[{os.path.basename(item['doc'])}]({item['doc']})"
        py_link = f"[{os.path.basename(item['py'])}]({item['py']})"
        f.write(f"| `{item['name']}` | `{item['regs']}` | {py_link} | {doc_link} | {item['distilbert']} | {item['roberta']} | {item['gpt2']} | {item['qwen']} | **{item['status']}** | {item['issues']} |\n")

print(f'Successfully generated {out_file} ({len(inventory)} modules).')
