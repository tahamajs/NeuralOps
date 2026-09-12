import os, re

with open('chapter_attentions.tex') as f:
    current_100 = [os.path.basename(l.split('{')[1].split('}')[0]).replace('.tex', '') for l in f if 'input{' in l]

bad_10 = {'weighted_ensemble', 'fused_svd_kernel', 'mamba_svd_flash', 'replace', 'samba_m3', 'samba_moe', 'spectral_mamba_attention', 'spectron_attention', 'svd_attention', 'svd_linear_decay'}

kept_90 = [m for m in current_100 if m not in bad_10]
print(f'Total kept pure passing modules: {len(kept_90)}')

# 1. Write updated chapter_attentions.tex
tex_lines = ['% Chapter 4: Monograph inputs for curated 90 pure-passing successful modules (Zero fails, zero zeros)\n']
for m in kept_90:
    tex_lines.append(f'\\input{{/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules/{m}.tex}}\n')

with open('chapter_attentions.tex', 'w', encoding='utf-8') as f:
    f.writelines(tex_lines)

# 2. Re-number headers in all 90 files
for idx, mod in enumerate(kept_90, 1):
    fpath = f'attentions_latex/individual_modules/{mod}.tex'
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    clean_mod = re.sub(r'[^a-zA-Z0-9]', '_', mod)
    escaped_mod = mod.replace('_', r'\_')

    # Replace subsection header number:
    pat = re.compile(r'\\subsection\{کالبدشکافی عمیق و تحلیل ساختاری ماژول شماره \d+:.*?\}')
    rep = f'\\subsection{{کالبدشکافی عمیق و تحلیل ساختاری ماژول شماره {idx}: \\lr{{{escaped_mod}}}}}'
    content = pat.sub(lambda _: rep, content)

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)

print('Updated chapter_attentions.tex and renumbered all 90 modules!')
