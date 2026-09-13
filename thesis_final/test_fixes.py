import glob
import re

files = ['master_thesis.tex', 'chapter_attentions.tex'] + glob.glob('attentions_latex/individual_modules/*.tex')

def is_math_or_command(s):
    if s.startswith('\\') or s.startswith('http') or s.startswith('fig:') or s.startswith('tab:') or s.startswith('sec:'):
        return True
    return False

# Test how many things would be fixed
total_p1_fixed = 0
total_p2_fixed = 0

sample_fixes = []

for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    
    in_math_block = False
    for line_idx, line in enumerate(lines):
        # check if in verbatim or math environment like equation
        if '\\begin{equation}' in line or '\\begin{align}' in line:
            in_math_block = True
        if '\\end{equation}' in line or '\\end{align}' in line:
            in_math_block = False
            continue
        if in_math_block:
            continue
        
        # Check Pattern 1: (\lr{...}) -> \lr{(...)}
        # E.g. (\lr{SVD}) -> \lr{(SVD)}
        matches_p1 = re.findall(r'\(\\lr\{([^}]+)\}\)', line)
        if matches_p1:
            total_p1_fixed += len(matches_p1)
            if len(sample_fixes) < 10:
                sample_fixes.append(('P1', line.strip()))

        # Check Pattern 2: (Latin text without Persian) not inside $...$
        # Match ( ... ) where inside has at least one ascii letter, NO Persian letters, and not already \lr{(...)}
        # We also need to ignore LaTeX commands like \ref{...}, \cite{...}
print(f"Total P1 (\\lr{{...}}) to fix: {total_p1_fixed}")
for kind, s in sample_fixes[:5]:
    print(f"Sample: {s}")
