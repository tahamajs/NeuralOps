import os, re

target_dir = '/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules'

for fname in os.listdir(target_dir):
    if not fname.endswith('.tex'): continue
    p = os.path.join(target_dir, fname)
    with open(p, encoding='utf-8') as f:
        content = f.read()
    
    # replace \& inside cases with &
    def fix_cases_match(m):
        cases_inner = m.group(1)
        cases_inner = cases_inner.replace(r'\&', '&')
        return r'\begin{cases}' + cases_inner + r'\end{cases}'
    
    new_content = re.sub(r'\\begin\{cases\}(.*?)\\end\{cases\}', fix_cases_match, content, flags=re.DOTALL)
    if new_content != content:
        print(f"Fixed cases in {fname}")
        with open(p, 'w', encoding='utf-8') as f:
            f.write(new_content)

