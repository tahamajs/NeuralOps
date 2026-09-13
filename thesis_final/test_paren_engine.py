import glob
import re

def fix_parentheses_in_text(text):
    # 1. First fix any existing (\lr{...}) to \lr{(...)}
    text = re.sub(r'\(\\lr\{([^}]+)\}\)', r'\\lr{(\1)}', text)
    
    # 2. Fix \lr{(\lr{...})} -> \lr{(...)}
    text = re.sub(r'\\lr\{\(\\lr\{([^}]+)\}\)\}', r'\\lr{(\1)}', text)

    # 3. Process line by line or by segments
    lines = text.split('\n')
    new_lines = []
    
    in_latin = False
    in_math_env = False
    
    for line in lines:
        if '\\begin{latin}' in line:
            in_latin = True
        if '\\end{latin}' in line:
            in_latin = False
            new_lines.append(line)
            continue
            
        if any(env in line for env in ['\\begin{equation}', '\\begin{align}', '\\begin{tikzpicture}', '\\begin{tabular}', '\\begin{lstlisting}']):
            in_math_env = True
        if any(env in line for env in ['\\end{equation}', '\\end{align}', '\\end{tikzpicture}', '\\end{tabular}', '\\end{lstlisting}']):
            in_math_env = False
            # We can still process tabular cells if carefully done, but let's check
        
        if in_latin:
            new_lines.append(line)
            continue
            
        # If line contains Persian text or section headings, let's fix bare Latin parentheses
        # Split line by inline math $...$ so we don't touch math formulas
        parts = re.split(r'(\$[^$]+\$)', line)
        for i in range(0, len(parts), 2): # even parts are outside $...$
            part = parts[i]
            
            # Pattern: (CONTENT) where CONTENT has ascii letters, no persian letters, no newlines, no other parens
            # and is NOT preceded by \lr, \ref, \cite, \label, \caption, \includegraphics
            def replace_paren(m):
                content = m.group(1).strip()
                # If already \lr or latex command, skip
                if content.startswith('\\lr') or content.startswith('\\ref') or content.startswith('\\cite') or content.startswith('\\label'):
                    return m.group(0)
                # If starts with backslash and has no spaces, like \alpha, skip
                if content.startswith('\\') and ' ' not in content:
                    return m.group(0)
                # Wrap in \lr{(...)}
                return f"\\lr{{({content})}}"
            
            # Match (?<!\\lr)(?<!\\label)(?<!\\ref)(?<!\\cite)\(([^()\u0600-\u06FF]*[a-zA-Z][^()\u0600-\u06FF]*)\)
            part = re.sub(r'(?<!\\lr)(?<!\\label)(?<!\\ref)(?<!\\cite)\(([^()\u0600-\u06FF]*[a-zA-Z][^()\u0600-\u06FF]*)\)', replace_paren, part)
            parts[i] = part
            
        new_lines.append(''.join(parts))
        
    return '\n'.join(new_lines)

# Test on a few module files
test_files = glob.glob('attentions_latex/individual_modules/*.tex')[:3]
for tf in test_files:
    with open(tf, 'r', encoding='utf-8') as fp:
        c = fp.read()
    new_c = fix_parentheses_in_text(c)
    # Print sample diffs
    diffs = [l for l in new_c.split('\n') if l not in c.split('\n')]
    print(f"=== {tf.split('/')[-1]} diff sample ({len(diffs)} changed lines) ===")
    for d in diffs[:4]:
        print("  ->", d[:100])
