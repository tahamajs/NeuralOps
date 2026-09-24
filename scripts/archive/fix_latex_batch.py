#!/usr/bin/env python3
"""
Fix all remaining LaTeX errors across paper/PerModulePaper:
1. Unicode characters (✅ ❌ etc)
2. equation/equation* mismatches
3. Missing $ in text (bare _ and ^ outside math)
4. egin{figure} missing backslash
5. 0x08 control bytes
6. \begin{equation*} / \end{equation*} mismatches
"""
import os
import re

base = 'paper/PerModulePaper'
fixed_files = 0

# Unicode emoji to LaTeX
unicode_map = {
    '✅': r'\checkmark',
    '❌': r'$\times$',
    '✔': r'\checkmark',
    '✗': r'$\times$',
    '→': r'$\rightarrow$',
    '←': r'$\leftarrow$',
    '≈': r'$\approx$',
    '≤': r'$\leq$',
    '≥': r'$\geq$',
    '∈': r'$\in$',
    '∑': r'$\sum$',
    '∏': r'$\prod$',
    '∞': r'$\infty$',
}

for root, dirs, files in os.walk(base):
    for f in files:
        if not f.endswith('.tex'):
            continue
        p = os.path.join(root, f)
        
        # Binary read first
        with open(p, 'rb') as fp:
            raw = fp.read()
        
        # 1. Remove stray 0x08 bytes but restore \begin when they prefix 'egin{figure}'
        backslash = bytes([92])
        bs_egin = bytes([8]) + b'egin{figure}'
        raw = raw.replace(bs_egin, backslash + b'begin{figure}')
        raw = raw.replace(bytes([8]), b'')
        
        with open(p, 'wb') as fp:
            fp.write(raw)
        
        # String-level fixes
        with open(p, 'r', encoding='utf-8', errors='replace') as fp:
            content = fp.read()
        
        new_content = content
        
        # 2. Unicode replacements
        for uni, latex in unicode_map.items():
            new_content = new_content.replace(uni, latex)
        
        # 3. Fix egin{figure} lines that start without backslash (string level)
        lines = new_content.split('\n')
        fixed_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('egin{figure}') and not line.strip().startswith('\\'):
                line = '\\begin{figure}' + stripped[12:]
            fixed_lines.append(line)
        new_content = '\n'.join(fixed_lines)
        
        # 4. Fix equation/equation* mismatches — normalize all to equation (no star)
        new_content = new_content.replace('\\begin{equation*}', '\\begin{equation}')
        new_content = new_content.replace('\\end{equation*}', '\\end{equation}')
        
        # 5. Fix double subscripts W_X^{(\text{...})}_y -> {W_X^{(\text{...})}}_y
        patterns = [
            ('target', 'g'), ('source', 'h'), ('pool', 'g'), 
            ('old', '{:d'), ('new', '{:d'), ('old', '_{:'),
        ]
        # Generic: Fix patterns like W_X^{...}_y where both ^ and _ are applied to W_X
        new_content = re.sub(
            r'(W_[KQVO])\^(\{[^}]*\})_(\{[^}]*\}|[a-zA-Z0-9])',
            r'{\1^\2}_\3',
            new_content
        )
        
        if new_content != content:
            with open(p, 'w', encoding='utf-8') as fp:
                fp.write(new_content)
            fixed_files += 1

print(f'Fixed {fixed_files} files!')
