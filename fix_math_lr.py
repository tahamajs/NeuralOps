import glob
import re

files = glob.glob('thesis_final/**/*.tex', recursive=True)
fixed_files = 0

for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        content = fp.read()
    
    # In progressive_token_pruning.tex
    # replace (\text{نشانه مرجع } \lr{[CLS]}) with \text{(نشانه مرجع [CLS])}
    new_content = content.replace(r'(\text{نشانه مرجع } \lr{[CLS]})', r'\text{(نشانه مرجع [CLS])}')
    
    # Also find any \lr{...} inside \begin{equation} ... \end{equation} that is not inside \text{...}
    # Pattern: \begin{equation} ... \end{equation}
    def clean_math(match):
        block = match.group(0)
        # replace standalone \lr{...} in math with \text{\lr{...}}
        # But only if not already preceded by \text{
        subbed = re.sub(r'(?<!\\text\{)\\lr\{([^}]+)\}', r'\\text{\\lr{\1}}', block)
        return subbed

    new_content = re.sub(r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}', clean_math, new_content, flags=re.DOTALL)
    new_content = re.sub(r'\\begin\{align\*?\}.*?\\end\{align\*?\}', clean_math, new_content, flags=re.DOTALL)

    if new_content != content:
        with open(f, 'w', encoding='utf-8') as fp:
            fp.write(new_content)
        fixed_files += 1

print(f'Fixed math lr in {fixed_files} files.')
