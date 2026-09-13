import glob
import os
import re

files = glob.glob('thesis_final/attentions_latex/individual_modules/*.tex')
fixed_count = 0

for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    
    new_lines = []
    changed = False
    in_tabular = False
    
    for line in lines:
        if r'\begin{tabular}' in line or r'\begin{matrix}' in line or r'\begin{align}' in line:
            in_tabular = True
        if r'\end{tabular}' in line or r'\end{matrix}' in line or r'\end{align}' in line:
            in_tabular = False
            
        if not in_tabular:
            # If line has section or caption or heading, replace & with \&
            if any(k in line for k in [r'\section', r'\subsection', r'\subsubsection', r'\caption', r'\paragraph']):
                # Replace unescaped & with \&
                new_line = re.sub(r'(?<!\\)&', r'\&', line)
                if new_line != line:
                    line = new_line
                    changed = True
        new_lines.append(line)
        
    if changed:
        with open(f, 'w', encoding='utf-8') as fp:
            fp.writelines(new_lines)
        fixed_count += 1

print(f'Fixed unescaped & in {fixed_count} files.')
