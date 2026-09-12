import glob
import re
import os

# 1. Gather all references from module files
module_files = sorted(glob.glob('attentions_latex/individual_modules/*.tex'))
print(f"Total module files found: {len(module_files)}")

raw_refs = []

# Pattern to find Section 12 block (matches \subsubsection*{بخش ۱۲: مراجع ...} or \subsection*{بخش ۱۲:...} to end of file)
sec12_pattern = re.compile(r'\n\s*\\(?:sub)*section\*?\{بخش ۱۲:[^\n]+.*$', re.DOTALL)

for f in module_files:
    with open(f, 'r', encoding='utf-8') as fp:
        content = fp.read()
    
    match = sec12_pattern.search(content)
    if match:
        sec12_text = match.group(0)
        # Extract \item ...
        items = re.findall(r'\\item\s+(.*?)(?=(?:\\item|\\end\{enumerate\}|\Z))', sec12_text, re.DOTALL)
        for it in items:
            cleaned = ' '.join(it.strip().split())
            if cleaned:
                raw_refs.append(cleaned)

print(f"Total raw references extracted from modules: {len(raw_refs)}")

# 2. Also extract existing references from master_thesis.tex
with open('master_thesis.tex', 'r', encoding='utf-8') as fp:
    master_content = fp.read()

m_master = re.search(r'\\chapter\*\{مراجع\}.*?\\begin\{enumerate\}(.*?)\\end\{enumerate\}', master_content, re.DOTALL)
if m_master:
    m_items = re.findall(r'\\item\s+(.*?)(?=(?:\\item|\Z))', m_master.group(1), re.DOTALL)
    for it in m_items:
        cleaned = ' '.join(it.strip().split())
        if cleaned:
            raw_refs.append(cleaned)

print(f"Total raw references including master_thesis: {len(raw_refs)}")

# 3. Clean and deduplicate references
def get_canonical_key(ref_text):
    # Strip LaTeX formatting
    s = re.sub(r'\\[a-zA-Z]+', '', ref_text)
    s = re.sub(r'[{}]', '', s)
    s = s.lower()
    # Extract year if possible
    year_match = re.search(r'\b(19\d\d|20\d\d)\b', s)
    year = year_match.group(1) if year_match else ""
    # Extract main alphanumeric keywords
    words = [w for w in re.findall(r'[a-z]+', s) if len(w) > 3 and w not in ['arxiv', 'preprint', 'proceedings', 'conference', 'journal', 'advances', 'learning', 'international', 'systems', 'neural', 'information']]
    key = (year, tuple(words[:4]))
    return key

# Prioritize more complete citations (longer strings with full details)
raw_refs.sort(key=lambda r: len(r), reverse=True)

deduped = []
seen_keys = set()

for r in raw_refs:
    # Ensure text is properly formatted (escape bare & to \& if not escaped)
    r_clean = re.sub(r'(?<!\\)&', r'\&', r)
    # Ensure braces are balanced
    if r_clean.count('{') != r_clean.count('}'):
        # simple balance check
        diff = r_clean.count('{') - r_clean.count('}')
        if diff > 0:
            r_clean += '}' * diff
    
    key = get_canonical_key(r_clean)
    if key not in seen_keys:
        seen_keys.add(key)
        deduped.append(r_clean)

print(f"Deduplicated references count: {len(deduped)}")

# Sort references alphabetically by first author's last name
def sort_key(ref):
    # Strip \textbf{ and get first letters
    s = re.sub(r'^\s*\\textbf\{\s*', '', ref)
    s = re.sub(r'[^a-zA-Z]', '', s)
    return s.lower()

deduped.sort(key=sort_key)

# 4. Update master_thesis.tex with all deduplicated references in the References chapter
# Build new References block
ref_items_tex = []
for r in deduped:
    ref_items_tex.append(f"    \\item {r}")

new_ref_list = "\n".join(ref_items_tex)

new_chapter_refs = f"""\\chapter*{{مراجع}}
\\addcontentsline{{toc}}{{chapter}}{{مراجع}}
\\begin{{latin}}
\\begin{{enumerate}}
{new_ref_list}
\\end{{enumerate}}
\\end{{latin}}"""

# Replace in master_thesis.tex
old_refs_pattern = re.compile(r'\\chapter\*\{مراجع\}.*?\\end\{latin\}', re.DOTALL)
if old_refs_pattern.search(master_content):
    master_updated = old_refs_pattern.sub(lambda m: new_chapter_refs, master_content)
    with open('master_thesis.tex', 'w', encoding='utf-8') as fp:
        fp.write(master_updated)
    print("Successfully updated master_thesis.tex with master bibliography!")
else:
    print("WARNING: Could not find references pattern in master_thesis.tex")

# 5. Remove Section 12 from all individual module files
stripped_count = 0
for f in module_files:
    with open(f, 'r', encoding='utf-8') as fp:
        content = fp.read()
    
    if sec12_pattern.search(content):
        # Strip Section 12 to end of file, leaving a clean trailing newline
        new_content = sec12_pattern.sub('', content).rstrip() + '\n'
        with open(f, 'w', encoding='utf-8') as fp:
            fp.write(new_content)
        stripped_count += 1

print(f"Cleanly stripped Section 12 from {stripped_count} module files.")
