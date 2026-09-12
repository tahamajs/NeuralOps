import glob
import re

def fix_line_parentheses(line):
    # 1. Fix (\lr{...}) -> \lr{(...)}
    # Handle multiple occurrences
    line = re.sub(r'\(\\lr\{([^}]+)\}\)', r'\\lr{(\1)}', line)
    
    # 2. Fix \lr{(\lr{...})} -> \lr{(...)} (avoid double nesting if any)
    line = re.sub(r'\\lr\{\(\\lr\{([^}]+)\}\)\}', r'\\lr{(\1)}', line)
    
    # 3. Known module headings
    headings_map = {
        '(Introduction)': r'\lr{(Introduction)}',
        '(Rationale)': r'\lr{(Rationale)}',
        '(Theoretical Foundations)': r'\lr{(Theoretical Foundations)}',
        '(Mathematical Derivation)': r'\lr{(Mathematical Derivation)}',
        '(Interpretation)': r'\lr{(Interpretation)}',
        '(Algorithmic Tensor Trace)': r'\lr{(Algorithmic Tensor Trace)}',
        '(PyTorch Implementation Details)': r'\lr{(PyTorch Implementation Details)}',
        '(Hardware Analysis)': r'\lr{(Hardware Analysis)}',
        '(Hardware & Roofline Analysis)': r'\lr{(Hardware \& Roofline Analysis)}',
        '(Hardware \& Roofline Analysis)': r'\lr{(Hardware \& Roofline Analysis)}',
        '(Empirical Benchmark & Scientific Analysis)': r'\lr{(Empirical Benchmark \& Scientific Analysis)}',
        '(Empirical Benchmark \& Scientific Analysis)': r'\lr{(Empirical Benchmark \& Scientific Analysis)}',
        '(Architectural Compatibility)': r'\lr{(Architectural Compatibility)}',
        '(Strengths & Weaknesses)': r'\lr{(Strengths \& Weaknesses)}',
        '(Strengths \& Weaknesses)': r'\lr{(Strengths \& Weaknesses)}',
        '(Section Conclusion)': r'\lr{(Section Conclusion)}',
        '(Figures & Charts)': r'\lr{(Figures \& Charts)}',
        '(Figures \& Charts)': r'\lr{(Figures \& Charts)}',
        '(TikZ Architecture Flowchart)': r'\lr{(TikZ Architecture Flowchart)}',
        '(References)': r'\lr{(References)}',
    }
    for k, v in headings_map.items():
        if k in line and ('\\section' in line or '\\subsection' in line or '\\subsubsection' in line):
            line = line.replace(k, v)
            
    return line

# Test on master_thesis.tex
with open('master_thesis.tex', 'r', encoding='utf-8') as fp:
    lines = fp.readlines()

fixed_count = 0
for idx, l in enumerate(lines):
    new_l = fix_line_parentheses(l)
    if new_l != l:
        fixed_count += 1
        if fixed_count <= 5:
            print(f"Original: {l.strip()}")
            print(f"Fixed:    {new_l.strip()}\n")

print(f"Total lines fixed in master_thesis.tex: {fixed_count}")
