import glob
import re

files = ['master_thesis.tex', 'chapter_attentions.tex'] + glob.glob('attentions_latex/individual_modules/*.tex')

# Pattern 1: (\lr{...}) -> Parentheses outside \lr
p1 = re.compile(r'\(\\lr\{([^}]+)\}\)')

# Pattern 2: bare english/latin/digits inside () without \lr
# e.g. (Rationale), (Theoretical Foundations), (N > 2048), (FP16/BF16), (2022)
p2 = re.compile(r'(?<!\\lr)\(([a-zA-Z0-9_ -]{2,})\)')

# Pattern 3: \lr{(...)} -> already properly wrapped inside \lr
p3 = re.compile(r'\\lr\{\(([^)]+)\)\}')

# Pattern 4: Persian text with latin inside ()
# e.g. (دیتاست SST-2) or (معماری ۱۲ لایه)
p4 = re.compile(r'\(([^)]*[\u0600-\u06FF][^)]*)\)')

c1, c2, c3, c4 = 0, 0, 0, 0
samples_p1 = []
samples_p2 = []

for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        content = fp.read()
    
    m1 = p1.findall(content)
    m2 = p2.findall(content)
    m3 = p3.findall(content)
    m4 = p4.findall(content)
    
    c1 += len(m1)
    c2 += len(m2)
    c3 += len(m3)
    c4 += len(m4)
    
    if m1 and len(samples_p1) < 5:
        samples_p1.extend(m1[:3])
    if m2 and len(samples_p2) < 5:
        samples_p2.extend(m2[:3])

print(f"Total files: {len(files)}")
print(f"Pattern 1 - (\\lr{{...}}) [OUTSIDE \\lr]: {c1} instances")
print(f"Sample Pattern 1: {samples_p1[:5]}")
print(f"Pattern 2 - (Bare Latin/Numbers): {c2} instances")
print(f"Sample Pattern 2: {samples_p2[:5]}")
print(f"Pattern 3 - \\lr{{(...)}} [CORRECT inside \\lr]: {c3} instances")
print(f"Pattern 4 - Persian inside (): {c4} instances")
