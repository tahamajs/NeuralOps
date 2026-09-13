import re
import glob
import os

files = glob.glob('thesis_final/attentions_latex/individual_modules/*.tex')

# A regex to match \text{...} where ... contains any English letter and doesn't already contain \lr{
# It should strictly capture the content inside \text{...}
# Because LaTeX can have nested braces, we use a simple approach assuming no nested braces inside \text{...} other than maybe what's already there.

def fix_text(match):
    inner = match.group(1)
    if re.search(r'[a-zA-Z]', inner) and '\\lr{' not in inner:
        # Wrap the whole inner thing in \lr{...}
        return f'\\text{{\\lr{{{inner}}}}}'
    return match.group(0)

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple regex for \text{something without braces}
    new_content = re.sub(r'\\text{([^{}]+)}', fix_text, content)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed {filepath}")
