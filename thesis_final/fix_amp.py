import os, re

target_dir = '/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules'

assigned = [
    'quantized_attention.tex', 'reformer_attention.tex', 'regla_attention.tex', 'replace.tex',
    'res_svd.tex', 'retention.tex', 'ring_attention.tex', 'routing_attention.tex',
    'salt.tex', 'salt_attention.tex', 'simple_ssm.tex', 'sinkhorn_attention.tex',
    'sliding_window_attention.tex', 'smart_weighted_ensemble.tex', 'soa_attn.tex',
    'soa_fused_module.tex', 'soa_module.tex', 'span_expanded_attention.tex',
    'spectra_v2.tex', 'spectral_attention.tex', 'spectron.tex', 'squeezing_heads.tex',
    'squeezing_heads_distill.tex', 'squeezing_heads_distiller.tex', 'structured_attention.tex',
    'structured_svd_attention.tex', 'svd_attention.tex', 'svd_compressed.tex'
]

for fname in assigned:
    p = os.path.join(target_dir, fname)
    with open(p, encoding='utf-8') as f:
        lines = f.readlines()
    
    in_tabular = False
    modified = False
    new_lines = []
    for line in lines:
        if '\\begin{tabular}' in line or '\\begin{array}' in line:
            in_tabular = True
        elif '\\end{tabular}' in line or '\\end{array}' in line:
            in_tabular = False
            
        if not in_tabular:
            # Look for & not preceded by \
            fixed_line = re.sub(r'(?<!\\)&', r'\\&', line)
            if fixed_line != line:
                print(f'{fname}: fixed unescaped & on: {line.strip()}')
                line = fixed_line
                modified = True
        new_lines.append(line)
        
    if modified:
        with open(p, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

print("Done checking unescaped &")
