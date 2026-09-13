# -*- coding: utf-8 -*-
"""
Fixes remaining syntax errors in master_thesis.tex and individual modules:
1. Replaces \pagenumbering{adad} with \pagenumbering{arabic}
2. Cleans duplicate braces in master_thesis.tex
3. Fixes stray braces in individual_modules/*.tex
"""

import re
import glob
import os

MASTER_FILE = "/Users/tahamajs/Documents/uni/Project/thesis_final/master_thesis.tex"
IM_DIR = "/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules"

# 1. Fix master_thesis.tex
with open(MASTER_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Fix pagenumbering
text = text.replace(r"\pagenumbering{adad}", r"\pagenumbering{arabic}")

# Fix double braces on \lr
# e.g. \lr{DistilBERT}} -> \lr{DistilBERT}
text = re.sub(r'(\\lr\{[^}]+\})\}+', r'\1', text)

# Fix items in list with extra brace: \item \textbf{\lr{DistilBERT}}}:
text = re.sub(r'(\\item\s*\\textbf\{\\lr\{[^}]+\}\}):\}*', r'\1:', text)

# Fix \textbf{\lr{Fusion Attention}})}
text = re.sub(r'(\\textbf\{[^\}]*\\lr\{[^\}]+\}[^\}]*)\}+', r'\1}', text)

# Clean any double backslash underscores
text = text.replace(r'\\_', r'\_')

with open(MASTER_FILE, "w", encoding="utf-8") as f:
    f.write(text)

print("master_thesis.tex cleaned.")

# 2. Fix smart_weighted_ensemble.tex and any other modules with stray braces
for fpath in glob.glob(os.path.join(IM_DIR, "*.tex")):
    with open(fpath, "r", encoding="utf-8") as fp:
        c = fp.read()

    # Fix / BERT} -> / \lr{BERT}
    c_new = re.sub(r'\\lr\{([^}]+)\}\s*/\s*BERT\}', lambda m: r'\lr{' + m.group(1) + r' / BERT}', c)
    c_new = re.sub(r'(\\lr\{[^}]+\})\}+', lambda m: m.group(1), c_new)

    if c_new != c:
        with open(fpath, "w", encoding="utf-8") as fp:
            fp.write(c_new)
        print(f"Fixed stray brace in {os.path.basename(fpath)}")

print("Module scan and fix complete.")
