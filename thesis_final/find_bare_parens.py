import glob
import re
from collections import Counter

files = ['master_thesis.tex', 'chapter_attentions.tex'] + glob.glob('attentions_latex/individual_modules/*.tex')

bare_parens = re.compile(r'(?<!\\lr)(?<!\\label)(?<!\\ref)(?<!\\cite)\(([^()\n\u0600-\u06FF]{2,})\)')

all_matches = []
for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        content = fp.read()
    
    # We want to skip \begin{latin} ... \end{latin}
    # and math blocks
    content_no_latin = re.sub(r'\\begin\{latin\}.*?\\end\{latin\}', '', content, flags=re.DOTALL)
    content_no_math = re.sub(r'\\begin\{(?:equation|align|tikzpicture)\*?\}.*?\\end\{(?:equation|align|tikzpicture)\*?\}', '', content_no_latin, flags=re.DOTALL)
    # Also remove inline math $...$
    content_no_inline_math = re.sub(r'\$[^$]+\$', '', content_no_math)
    
    for m in bare_parens.finditer(content_no_inline_math):
        text = m.group(1).strip()
        # ignore if it's already an lr or command
        if not text.startswith('\\lr') and not text.startswith('\\ref') and not text.startswith('\\label'):
            all_matches.append(text)

print(f"Total bare Latin/number parentheses found: {len(all_matches)}")
c = Counter(all_matches)
print("Top 35 most common bare parentheses:")
for item, cnt in c.most_common(35):
    print(f"  ({item}) : {cnt}")
