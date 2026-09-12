import glob
import re

files = ['master_thesis.tex', 'chapter_attentions.tex'] + glob.glob('attentions_latex/individual_modules/*.tex')

# Find all occurrences of (CONTENT) where CONTENT is ASCII/digits and not inside \lr
bare_parens = re.compile(r'(?<!\\lr)(?<!\\label)(?<!\\ref)(?<!\\cite)\(([^()\n\u0600-\u06FF]*[a-zA-Z0-9][^()\n\u0600-\u06FF]*)\)')

files_with_bare = {}
for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        c = fp.read()
    # strip environments
    c_clean = re.sub(r'\\begin\{latin\}.*?\\end\{latin\}', '', c, flags=re.DOTALL)
    c_clean = re.sub(r'\\begin\{(?:equation|align|gather|multline|tikzpicture|lstlisting)\*?\}.*?\\end\{(?:equation|align|gather|multline|tikzpicture|lstlisting)\*?\}', '', c_clean, flags=re.DOTALL)
    c_clean = re.sub(r'\$\$[^\$]+\$\$', '', c_clean, flags=re.DOTALL)
    c_clean = re.sub(r'\$[^\$]+\$', '', c_clean)
    c_clean = re.sub(r'\\lr\{[^{}]*\}', '', c_clean)
    
    matches = bare_parens.findall(c_clean)
    # filter out commands like \ref, \label, etc.
    real_matches = [m.strip() for m in matches if not m.strip().startswith('\\') and not m.strip().startswith('http') and not m.strip().startswith('XXPH')]
    if real_matches:
        files_with_bare[f] = real_matches

print(f"Files with bare parentheses in Persian text: {len(files_with_bare)}")
for f, m in list(files_with_bare.items())[:10]:
    print(f"{f.split('/')[-1]}: {len(m)} matches -> {m[:5]}")
