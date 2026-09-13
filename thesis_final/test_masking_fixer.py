import glob
import re

def fix_parentheses_robust(text):
    # 1. First, fix explicit (\lr{...}) to \lr{(...)}
    text = re.sub(r'\(\\lr\{([^}]+)\}\)', r'\\lr{(\1)}', text)

    # 2. Mask environments: latin, equations, tikz, tabular
    placeholders = []
    
    def add_placeholder(s):
        key = f"XXPH{len(placeholders)}ZZ"
        placeholders.append((key, s))
        return key
    
    # Mask \begin{latin}...\end{latin}
    text = re.sub(r'\\begin\{latin\}.*?\\end\{latin\}', lambda m: add_placeholder(m.group(0)), text, flags=re.DOTALL)
    
    # Mask multiline math environments
    text = re.sub(r'\\begin\{(?:equation|align|gather|multline|tikzpicture|lstlisting)\*?\}.*?\\end\{(?:equation|align|gather|multline|tikzpicture|lstlisting)\*?\}', 
                  lambda m: add_placeholder(m.group(0)), text, flags=re.DOTALL)
    
    # Mask display math $$...$$
    text = re.sub(r'\$\$[^\$]+\$\$', lambda m: add_placeholder(m.group(0)), text, flags=re.DOTALL)

    # Mask inline math $...$
    text = re.sub(r'\$[^\$]+\$', lambda m: add_placeholder(m.group(0)), text)
    
    # Mask labels, refs, cites, includegraphics
    text = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?\{[^}]*\}', 
                  lambda m: add_placeholder(m.group(0)) if any(k in m.group(0) for k in ['\\ref', '\\label', '\\cite', '\\includegraphics', '\\url']) else m.group(0), text)

    # Mask existing \lr{...}
    def mask_lr(match_text):
        res = []
        pos = 0
        pattern = re.compile(r'\\lr\{')
        while True:
            m = pattern.search(match_text, pos)
            if not m:
                res.append(match_text[pos:])
                break
            res.append(match_text[pos:m.start()])
            # find matching brace
            brace_count = 1
            i = m.end()
            while i < len(match_text) and brace_count > 0:
                if match_text[i] == '{':
                    brace_count += 1
                elif match_text[i] == '}':
                    brace_count -= 1
                i += 1
            lr_content = match_text[m.start():i]
            res.append(add_placeholder(lr_content))
            pos = i
        return ''.join(res)

    text = mask_lr(text)

    # 3. Now find all bare parentheses in Persian text
    # Match (CONTENT) where CONTENT has no Persian characters, has at least one ASCII letter or digit, and no placeholders
    def replace_bare_paren(m):
        content = m.group(1).strip()
        # If it contains placeholder, do not touch
        if 'XXPH' in content:
            return m.group(0)
        # Wrap as \lr{(content)}
        return f"\\lr{{({content})}}"

    # ( ... ) without Persian letters [\u0600-\u06FF] and without newlines or other parens
    text = re.sub(r'\(([^()\n\u0600-\u06FF]*[a-zA-Z0-9][^()\n\u0600-\u06FF]*)\)', replace_bare_paren, text)

    # 4. Restore placeholders in reverse order
    for key, val in reversed(placeholders):
        text = text.replace(key, val)
    
    # 5. Clean up any accidental double \lr{\lr{(...)}} -> \lr{(...)}
    text = re.sub(r'\\lr\{\\lr\{(\([^}]+\))\bottom\}', r'\\lr{\1}', text)
    text = re.sub(r'\\lr\{\\lr\{([^}]+)\}\}', r'\\lr{\1}', text)

    return text

# Test on sample files
test_files = [
    'attentions_latex/individual_modules/hla_attention.tex',
    'attentions_latex/individual_modules/soa_fused_module.tex',
    'attentions_latex/individual_modules/dynamic_sparse_attention.tex',
    'master_thesis.tex'
]

for tf in test_files:
    with open(tf, 'r', encoding='utf-8') as fp:
        c = fp.read()
    new_c = fix_parentheses_robust(c)
    diffs = [l.strip() for l in new_c.split('\n') if l.strip() not in [x.strip() for x in c.split('\n')]]
    print(f"\n=== {tf.split('/')[-1]} ({len(diffs)} changed lines) ===")
    for d in diffs[:6]:
        print("  ->", d[:110])
