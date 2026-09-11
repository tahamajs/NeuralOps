import subprocess, os

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

test_tex_template = r'''\documentclass[12pt,a4paper]{report}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{color,colortbl}
\usepackage{booktabs}
\usepackage{tikz}
\usetikzlibrary{arrows,positioning,shapes,calc,fit,matrix}
\usepackage{hyperref}
\usepackage{xepersian}
\settextfont{X Nazanin}
\setdigitfont{X Nazanin}
\setlatintextfont{Times New Roman}

\makeatletter
\def\UseMathForPositioningText{}
\def\@kernel@tabular@init{}
\makeatother

\begin{document}
%s
\end{document}
'''

failed = []
for fname in assigned:
    inc = f'\\input{{attentions_latex/individual_modules/{fname}}}'
    test_src = test_tex_template % inc
    with open('/Users/tahamajs/Documents/uni/Project/thesis_final/test_single.tex', 'w', encoding='utf-8') as f:
        f.write(test_src)
    res = subprocess.run(['xelatex', '-interaction=nonstopmode', 'test_single.tex'], cwd='/Users/tahamajs/Documents/uni/Project/thesis_final', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f'COMPILE FAILED: {fname}')
        failed.append(fname)
    else:
        print(f'OK: {fname}')

if not failed:
    print('ALL 28 FILES COMPILED PERFECTLY WITH EXIT CODE 0!')
else:
    print(f'FAILED FILES: {failed}')

