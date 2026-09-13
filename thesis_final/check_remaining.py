import glob
import re

files = ['master_thesis.tex', 'chapter_attentions.tex'] + glob.glob('attentions_latex/individual_modules/*.tex')
p_lr = re.compile(r'\(\\lr\{([^}]+)\}\)')
# Lookbehind that properly checks NOT preceded by \lr{
p_bare = re.compile(r'(?<!\\lr\{)(?<!\\lr\[)(?<!\\cite\{)(?<!\\ref\{)(?<!\\label\{)\(([^()\n\u0600-\u06FF]*[a-zA-Z][^()\n\u0600-\u06FF]*)\)')

count_lr = 0
count_bare = 0
sample_bare = []

for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        c = fp.read()
    c = re.sub(r'\\begin\{latin\}.*?\\end\{latin\}', '', c, flags=re.DOTALL)
    c = re.sub(r'\$[^\$]+\$', '', c)
    count_lr += len(p_lr.findall(c))
    m = p_bare.findall(c)
    real_m = [x for x in m if not x.strip().startswith('\\') and not x.strip().startswith('http')]
    count_bare += len(real_m)
    if real_m and len(sample_bare) < 5:
        sample_bare.append((f.split('/')[-1], real_m[:3]))

print(f"Remaining (\\lr{{...}}): {count_lr}")
print(f"Remaining bare Latin in parens (without \\lr{{...}}): {count_bare}")
for item in sample_bare:
    print(item)
