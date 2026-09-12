import glob
import re

files = ['master_thesis.tex', 'chapter_attentions.tex'] + glob.glob('attentions_latex/individual_modules/*.tex')

# Let's inspect all kinds of problematic parenthesis patterns:

# 1. ($MATH$) -> Parens outside math: e.g. ($N > 2048$)
p_math_outside = re.compile(r'\(\$([^$]+)\$\)')

# 2. (\lr{LATIN}) -> Parens outside \lr: e.g. (\lr{SVD})
p_lr_outside = re.compile(r'\(\\lr\{([^}]+)\}\)')

# 3. (BARE_LATIN) -> Bare latin words in parentheses in Persian text: e.g. (MoE) or (Rationale)
# outside \lr and outside math
p_bare_latin = re.compile(r'(?<!\\lr)(?<!\\cite)(?<!\\ref)(?<!\\label)\(([^()\n\u0600-\u06FF]*[a-zA-Z][^()\n\u0600-\u06FF]*)\)')

# 4. Bare digits in parentheses: e.g. (2022) or (2017) or (1984) in Persian text
p_bare_year = re.compile(r'(?<!\\lr)(?<!\\cite)(?<!\\ref)(?<!\\label)\(((?:19|20)\d\d)\)')

total_math = 0
total_lr = 0
total_bare = 0
total_year = 0

sample_math = []
sample_lr = []
sample_bare = []
sample_year = []

for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        c = fp.read()
    
    # Exclude \begin{latin}...\end{latin}
    c_persian = re.sub(r'\\begin\{latin\}.*?\\end\{latin\}', '', c, flags=re.DOTALL)
    
    m1 = p_math_outside.findall(c_persian)
    m2 = p_lr_outside.findall(c_persian)
    m3 = p_bare_latin.findall(c_persian)
    m4 = p_bare_year.findall(c_persian)
    
    total_math += len(m1)
    total_lr += len(m2)
    total_bare += len(m3)
    total_year += len(m4)
    
    if m1 and len(sample_math) < 5:
        sample_math.extend(m1[:3])
    if m2 and len(sample_lr) < 5:
        sample_lr.extend(m2[:3])
    if m3 and len(sample_bare) < 5:
        sample_bare.extend(m3[:3])
    if m4 and len(sample_year) < 5:
        sample_year.extend(m4[:3])

print(f"1. Parens outside math ($...$): {total_math} instances. Samples: {sample_math[:5]}")
print(f"2. Parens outside \\lr (\\lr{{...}}): {total_lr} instances. Samples: {sample_lr[:5]}")
print(f"3. Bare Latin inside parens: {total_bare} instances. Samples: {sample_bare[:5]}")
print(f"4. Bare Year inside parens (YYYY): {total_year} instances. Samples: {sample_year[:5]}")
