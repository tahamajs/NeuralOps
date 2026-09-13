import glob
import re
import os

files = ['master_thesis.tex', 'chapter_attentions.tex'] + glob.glob('attentions_latex/individual_modules/*.tex')
print(f"Total files to process: {len(files)}")

def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    
    new_lines = []
    
    in_latin_block = False
    in_math_block = False
    
    for line in lines:
        stripped = line.strip()
        
        # Track blocks
        if '\\begin{latin}' in line:
            in_latin_block = True
        if '\\end{latin}' in line:
            in_latin_block = False
            new_lines.append(line)
            continue
            
        if any(env in line for env in ['\\begin{equation}', '\\begin{align}', '\\begin{gather}', '\\begin{multline}', '\\begin{tikzpicture}', '\\begin{lstlisting}']):
            in_math_block = True
        if any(env in line for env in ['\\end{equation}', '\\end{align}', '\\end{gather}', '\\end{multline}', '\\end{tikzpicture}', '\\end{lstlisting}']):
            in_math_block = False
            new_lines.append(line)
            continue
            
        if in_latin_block or in_math_block:
            new_lines.append(line)
            continue
            
        # Step 1: Fix unclosed braces on section heading lines
        if re.search(r'\\(?:sub)*section\*?\{', line):
            # Check if braces are unbalanced
            diff = line.count('{') - line.count('}')
            if diff > 0:
                # Add missing closing braces before newline
                line = line.rstrip('\r\n') + ('}' * diff) + '\n'

        # Step 2: Fix ($...$) -> $(...)$ where parens are outside math
        # e.g. ($N > 2048$) -> $(N > 2048)$
        # e.g. ($\mathbf{Q}$) -> $(\mathbf{Q})$
        line = re.sub(r'\(\$([^$]+)\$\)', r'$(\1)$', line)
        
        # Step 3: Fix (\lr{...}) -> \lr{(...)}
        line = re.sub(r'\(\\lr\{([^}]+)\}\)', r'\\lr{(\1)}', line)
        
        # Step 4: Fix bare Latin in parentheses outside \lr and outside $...$
        # e.g. (Locality Sensitive Hashing) -> \lr{(Locality Sensitive Hashing)}
        # e.g. (Roofline Model & IO Complexity) -> \lr{(Roofline Model \& IO Complexity)}
        # Split by math $...$
        parts = re.split(r'(\$[^$]+\$)', line)
        for i in range(0, len(parts), 2): # even parts are outside $...$
            p = parts[i]
            
            # Match (CONTENT) where CONTENT has ascii letters, no persian letters, no other parens
            # and is not preceded by \lr, \ref, \label, \cite
            def replace_bare(m):
                content = m.group(1).strip()
                # If already \lr or latex command, skip
                if content.startswith('\\lr') or content.startswith('\\ref') or content.startswith('\\label') or content.startswith('\\cite'):
                    return m.group(0)
                # If it's a bare command like \alpha, skip
                if content.startswith('\\') and ' ' not in content:
                    return m.group(0)
                # Escape unescaped & to \&
                content_esc = re.sub(r'(?<!\\)&', r'\&', content)
                return f"\\lr{{({content_esc})}}"
            
            # Pattern: (?<!\\lr\{)(?<!\\lr)(?<!\\ref)(?<!\\label)(?<!\\cite)\(([^()\u0600-\u06FF]*[a-zA-Z][^()\u0600-\u06FF]*)\)
            p = re.sub(r'(?<!\\lr\{)(?<!\\lr)(?<!\\ref)(?<!\\label)(?<!\\cite)\(([^()\n\u0600-\u06FF]*[a-zA-Z][^()\n\u0600-\u06FF]*)\)', replace_bare, p)
            parts[i] = p
            
        line = ''.join(parts)
        
        # Step 5: Clean any accidental double nesting: \lr{\lr{(...)}} -> \lr{(...)}
        line = re.sub(r'\\lr\{\\lr\{(\([^}]+\))\bottom\}', r'\\lr{\1}', line)
        line = re.sub(r'\\lr\{\\lr\{([^}]+)\}\}', r'\\lr{\1}', line)
        line = re.sub(r'\\lr\{\(\\lr\{([^}]+)\}\)\}', r'\\lr{(\1)}', line)
        
        new_lines.append(line)
        
    new_content = ''.join(new_lines)
    with open(filepath, 'w', encoding='utf-8') as fp:
        fp.write(new_content)

changed_files = 0
for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        orig = fp.read()
    fix_file(f)
    with open(f, 'r', encoding='utf-8') as fp:
        updated = fp.read()
    if updated != orig:
        changed_files += 1

print(f"Successfully processed and updated {changed_files} files!")
