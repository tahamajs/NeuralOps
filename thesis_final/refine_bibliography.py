import re

with open('master_thesis.tex', 'r', encoding='utf-8') as fp:
    content = fp.read()

m = re.search(r'\\chapter\*\{مراجع\}.*?\\begin\{enumerate\}(.*?)\\end\{enumerate\}', content, re.DOTALL)
if not m:
    print("Could not find references block")
    exit(1)

items = re.findall(r'\\item\s+(.*?)(?=(?:\\item|\Z))', m.group(1), re.DOTALL)
clean_items = [' '.join(it.strip().split()) for it in items if it.strip()]

# Function to extract title keyword
def extract_title_key(ref):
    # Match italic title or words after year
    m_it = re.search(r'\\textit\{([^}]+)\}', ref)
    # Also find text between ).} and \textit
    m_title = re.search(r'\(\d{4}\)\.\}\s*([^.]+)\.', ref)
    if m_title:
        title = m_title.group(1).lower()
    elif m_it:
        title = m_it.group(1).lower()
    else:
        title = ref.lower()
    
    # Keep only letters
    title_slug = re.sub(r'[^a-z]', '', title)[:30]
    return title_slug

# Group items by title_slug
groups = {}
for it in clean_items:
    slug = extract_title_key(it)
    if slug not in groups:
        groups[slug] = []
    groups[slug].append(it)

# For each group, pick the best reference (prefer non-et al, and longer string)
final_refs = []
for slug, candidates in groups.items():
    # Sort: prefer not having 'et al', then prefer longer length
    def score(c):
        has_etal = 'et al' in c.lower()
        return (not has_etal, len(c))
    
    candidates.sort(key=score, reverse=True)
    final_refs.append(candidates[0])

# Alphabetize by primary author last name
def get_author_key(ref):
    clean = re.sub(r'^\s*\\textbf\{\s*', '', ref)
    return clean.lower()

final_refs.sort(key=get_author_key)

print(f"Original items: {len(clean_items)}, Unique canonical items: {len(final_refs)}")

new_items_tex = "\n".join([f"    \\item {r}" for r in final_refs])

new_block = f"""\\chapter*{{مراجع}}
\\addcontentsline{{toc}}{{chapter}}{{مراجع}}
\\begin{{latin}}
\\begin{{enumerate}}
{new_items_tex}
\\end{{enumerate}}
\\end{{latin}}"""

pattern = re.compile(r'\\chapter\*\{مراجع\}.*?\\end\{latin\}', re.DOTALL)
updated_content = pattern.sub(lambda m: new_block, content)

with open('master_thesis.tex', 'w', encoding='utf-8') as fp:
    fp.write(updated_content)

print("master_thesis.tex successfully updated with clean canonical bibliography!")
