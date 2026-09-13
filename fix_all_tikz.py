import glob
import re

files = glob.glob('thesis_final/attentions_latex/individual_modules/*.tex')
fixed_files = 0

for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        c = fp.read()
    
    if r'\begin{tikzpicture}' in c:
        # In tikzpicture styles, ensure text centered has align=center
        # e.g. text centered, -> align=center, text centered,
        # but only if align= is not already present in that style definition
        new_c = c
        # Replace 'text centered' with 'align=center, text centered' if not preceded by align=
        new_c = re.sub(r'(?<!align=center,\s)text centered', 'align=center, text centered', new_c)
        # remove any accidental double 'align=center, align=center'
        new_c = new_c.replace('align=center, align=center,', 'align=center,')
        
        if new_c != c:
            with open(f, 'w', encoding='utf-8') as fp:
                fp.write(new_c)
            fixed_files += 1

print(f'Fixed TikZ styles in {fixed_files} files.')
