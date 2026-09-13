import glob
import re

files = glob.glob('thesis_final/**/*.tex', recursive=True)
fixed_files = 0

for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        c = fp.read()
    
    # Match any \lr{...} and escape any unescaped underscore inside it
    def fix_lr(match):
        inner = match.group(1)
        escaped = re.sub(r'(?<!\\)_', r'\\_', inner)
        return f'\\lr{{{escaped}}}'

    new_c = re.sub(r'\\lr\{([^}]+)\}', fix_lr, c)
    
    if new_c != c:
        with open(f, 'w', encoding='utf-8') as fp:
            fp.write(new_c)
        fixed_files += 1

print(f'Fixed unescaped underscores in \\lr{{}} across {fixed_files} files.')
