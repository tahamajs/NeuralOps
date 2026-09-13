import os
import glob
import re

directory = 'thesis_final/attentions_latex/individual_modules'
files = glob.glob(os.path.join(directory, '*.tex'))

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # We want to replace (\lr{module}) with \lr{(module)}
    
    content = re.sub(
        r'\(\\lr\{([^}]+)\}\)',
        r'\\lr{(\1)}',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

print("Updated parentheses in captions.")
