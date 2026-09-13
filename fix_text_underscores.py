import glob
import re

files = glob.glob('thesis_final/attentions_latex/individual_modules/*.tex')
fixed_count = 0

for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        c = fp.read()
    
    # Replace \text{something_something} where _ is not escaped
    def fix_text(match):
        inner = match.group(1)
        # Escape any unescaped underscore
        escaped = re.sub(r'(?<!\\)_', r'\\_', inner)
        return f'\\text{{{escaped}}}'

    new_c = re.sub(r'\\text\{([^}]+)\}', fix_text, c)
    
    if new_c != c:
        with open(f, 'w', encoding='utf-8') as fp:
            fp.write(new_c)
        fixed_count += 1

print(f'Fixed unescaped underscores in \\text{{}} across {fixed_count} files.')
