import glob
import re

files = ['master_thesis.tex', 'chapter_attentions.tex'] + glob.glob('attentions_latex/individual_modules/*.tex')

# Let's inspect headings in modules
heading_patterns = [
    (r'(\\\w+\*?\{[^}]*)\(Rationale\)([^}]*\})', r'\1\\lr{(Rationale)}\2'),
    (r'(\\\w+\*?\{[^}]*)\(Theoretical Foundations\)([^}]*\})', r'\1\\lr{(Theoretical Foundations)}\2'),
    (r'(\\\w+\*?\{[^}]*)\(Mathematical Derivation\)([^}]*\})', r'\1\\lr{(Mathematical Derivation)}\2'),
    (r'(\\\w+\*?\{[^}]*)\(PyTorch Implementation Details\)([^}]*\})', r'\1\\lr{(PyTorch Implementation Details)}\2'),
    (r'(\\\w+\*?\{[^}]*)\(Hardware Analysis\)([^}]*\})', r'\1\\lr{(Hardware Analysis)}\2'),
    (r'(\\\w+\*?\{[^}]*)\(Empirical Benchmark & Scientific Analysis\)([^}]*\})', r'\1\\lr{(Empirical Benchmark & Scientific Analysis)}\2'),
    (r'(\\\w+\*?\{[^}]*)\(Architectural Compatibility\)([^}]*\})', r'\1\\lr{(Architectural Compatibility)}\2'),
    (r'(\\\w+\*?\{[^}]*)\(Strengths & Weaknesses\)([^}]*\})', r'\1\\lr{(Strengths & Weaknesses)}\2'),
    (r'(\\\w+\*?\{[^}]*)\(Section Conclusion\)([^}]*\})', r'\1\\lr{(Section Conclusion)}\2'),
    (r'(\\\w+\*?\{[^}]*)\(Figures & Charts\)([^}]*\})', r'\1\\lr{(Figures & Charts)}\2'),
    (r'(\\\w+\*?\{[^}]*)\(References\)([^}]*\})', r'\1\\lr{(References)}\2'),
]

total_matched = 0
for pat, rep in heading_patterns:
    regex = re.compile(pat)
    count = 0
    for f in files:
        with open(f, 'r', encoding='utf-8') as fp:
            c = fp.read()
        count += len(regex.findall(c))
    print(f"Matched {count} for {pat[:30]}...")
    total_matched += count

print(f"Total section heading parentheses to fix: {total_matched}")
