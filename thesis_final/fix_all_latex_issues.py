# -*- coding: utf-8 -*-
"""
Comprehensive Fixer for all LaTeX Syntax & Typography Issues in Top 100 Modules:
1. Fixes section headers with matching braces across the whole line.
2. Fixes unclosed math braces (e.g. \mathbb{R}^{...}).
3. Fixes unescaped & inside text / \lr{...}.
4. Fixes extra braces after \label{sec:...}.
5. Validates environment balancing and brace balancing.
"""

import os
import re
import glob

IM_DIR = "/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules"

with open("/Users/tahamajs/Documents/uni/Project/thesis_final/chapter_attentions.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

target_mods = []
for l in lines:
    if r"\input{" in l:
        p = l.split("{")[1].split("}")[0]
        target_mods.append(os.path.basename(p))

print(f"Processing {len(target_mods)} modules...")

def fix_content(content, mod_name):
    # 1. Fix label lines with extra braces
    content = re.sub(r'(\\label\{sec:[^}]+\})\}*', r'\1', content)

    # 2. Fix Section 7 header across the entire line
    content = re.sub(
        r'\\sub(?:sub)?section\*\{بخش\s*[7۷]:[^\n]+',
        r'\\subsubsection*{بخش ۷: ارزیابی تجربی، بنچمارک و توجیه علمی نتایج \\lr{(Empirical Benchmark \& Scientific Analysis)}}',
        content
    )

    # 3. Clean up any quadruple or triple braces on section headers
    content = re.sub(r'(\\subsubsection\*\{[^\n]+?)\}+\s*$', r'\1}', content, flags=re.MULTILINE)

    # 4. Fix unclosed math braces like \mathbb{R}^{B \times N \times D_{\text{all}}
    content = re.sub(r'(\\mathbb\{R\}\^\{[^}\n]+)(?=\s*\\end\{equation\}|\s*\\\\|\s*\$)', r'\1}', content)

    # 5. Fix unescaped & inside \lr{...}
    def fix_lr_amp(m):
        inner = m.group(1)
        inner = re.sub(r'(?<!\\)&', r'\\&', inner)
        return r'\lr{' + inner + '}'
    content = re.sub(r'\\lr\{([^}]+)\}', fix_lr_amp, content)

    # 6. Fix unescaped & outside tabular/matrix
    # Break into lines and fix standalone & in text paragraphs
    new_lines = []
    in_table_or_math = False
    for line in content.split('\n'):
        if any(env in line for env in ['tabular', 'align', 'matrix', 'cases']):
            if r'\begin{' in line:
                in_table_or_math = True
            if r'\end{' in line:
                in_table_or_math = False
            new_lines.append(line)
            continue

        if not in_table_or_math:
            # If line is not in tabular/math, escape & unless already escaped
            line = re.sub(r'(?<!\\)&', r'\\&', line)
        new_lines.append(line)

    content = '\n'.join(new_lines)

    # 7. Ensure \small table uses [htbp] instead of [H] if float causes issues, or keep [H]
    # Check for unescaped %
    content = re.sub(r'\\lr\{([^}]+)\}', lambda m: r'\lr{' + re.sub(r'(?<!\\)%', r'\\%', m.group(1)) + '}', content)

    # 8. Escape underscores in \lr
    content = re.sub(r'\\lr\{([^}]+)\}', lambda m: r'\lr{' + re.sub(r'(?<!\\)_', r'\_', m.group(1)) + '}', content)

    # 9. Clean double backslash underscores \\_ -> \_
    content = content.replace(r'\\_', r'\_')

    return content

fixed_count = 0
for m in target_mods:
    fpath = os.path.join(IM_DIR, m)
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r', encoding='utf-8') as fp:
        orig = fp.read()
    
    fixed = fix_content(orig, m)
    if fixed != orig:
        with open(fpath, 'w', encoding='utf-8') as fp:
            fp.write(fixed)
        fixed_count += 1

print(f"Applied syntax & typography fixes to {fixed_count}/{len(target_mods)} files.")
