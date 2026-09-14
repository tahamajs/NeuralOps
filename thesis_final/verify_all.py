import os
import re

base = '/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules'

modules = [
    'qcqa_attention',
    'rala_attention',
    'rala_compressor',
    'reap_moe',
    'regla_attention',
    'replace',
    'res_svd',
    'routing_attention',
    'rpc',
    'rpc_attention',
    'salt',
    'salt_attention',
    'samba_m3',
    'samba_moe',
    'simple_ssm',
    'smart_weighted_ensemble',
    'soa_attn',
    'span_expanded_attention',
    'spectra_v2',
    'spectra_v2_attention',
    'spectral_mamba',
    'spectral_mamba_attention',
    'spectron',
    'spectron_attention',
    'squeezing_heads',
    'squeezing_heads_distill',
    'squeezing_heads_distiller',
    'svd_attention',
    'svd_compressed'
]

print(f'Verifying {len(modules)} modules...')
all_ok = True

for m in modules:
    p = os.path.join(base, m + '.tex')
    if not os.path.exists(p):
        # Some legacy generated inputs intentionally omit the .tex suffix;
        # TeX accepts both forms, so the verifier must follow the same rule.
        p = os.path.join(base, m)
    with open(p, encoding='utf-8') as f:
        content = f.read()
    
    # 1. Check section header
    if not re.search(r'\\section\{کالبدشکافی ماژول \\lr\{[^}]+\}\}', content):
        print(f'[{m}] Section header check FAILED')
        all_ok = False
        
    # 2. Check TikZ picture
    if r'\begin{tikzpicture}' not in content or r'\end{tikzpicture}' not in content:
        print(f'[{m}] TikZ check FAILED')
        all_ok = False
        
    # 3. Check benchmarking table
    if r'\begin{tabular}' not in content or r'\end{tabular}' not in content:
        print(f'[{m}] Tabular check FAILED')
        all_ok = False
        
    # 4. Check for double escaped underscores \\_
    if r'\\_' in content:
        print(f'[{m}] Found double-escaped underscore \\\\_')
        all_ok = False
        
    # 5. Check for unescaped underscores inside \lr{...}
    lr_bad = re.findall(r'\\lr\{[^}]*(?<!\\)_[^}]*\}', content)
    if lr_bad:
        print(f'[{m}] Found unescaped underscore inside \\lr: {lr_bad}')
        all_ok = False
        
    # 6. Check for English % in text (not comments)
    for line_no, line in enumerate(content.splitlines()):
        stripped = line.strip()
        if stripped.startswith('%'):
            continue
        match = re.search(r'(\d+)\s*%', line)
        if match:
            print(f'[{m}] Line {line_no+1}: Found English %: "{line}"')
            all_ok = False

if all_ok:
    print('ALL 29 FILES PASSED VERIFICATION PERFECTLY!')
else:
    print('Some checks failed.')
