import glob
import re

files = glob.glob('attentions_latex/individual_modules/*.tex')
print(f"Total files: {len(files)}")

# Pattern to find Section 12 (بخش ۱۲: مراجع)
# Matches \subsubsection*{بخش ۱۲: مراجع ...} or \subsection*{بخش ۱۲: مراجع ...} to the end of file
pattern = re.compile(r'\n\s*\\(?:sub)*section\*?\{بخش ۱۲:[^\n]+.*$', re.DOTALL)

matched = 0
for f in sorted(files):
    with open(f, 'r', encoding='utf-8') as fp:
        content = fp.read()
    if pattern.search(content):
        matched += 1

print(f"Found Section 12 in {matched} files.")
