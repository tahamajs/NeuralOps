import glob
import re

files = sorted(glob.glob('/Users/tahamajs/Documents/uni/Project/Docs/content/attentions/*.md') + glob.glob('/Users/tahamajs/Documents/uni/Project/Docs/content/compressions/*.md'))

print(f"Total markdown files: {len(files)}")
diagram_blocks = []

for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        c = fp.read()
    blocks = re.findall(r'```(.*?)```', c, re.DOTALL)
    for idx, b in enumerate(blocks):
        b_clean = b.strip()
        # check if it's an ASCII diagram
        if any(char in b_clean for char in ['┌', '─', '│', '▼', '▲', '──►', '+---', '├──', '└──', '---|']):
            if not any(kw in b_clean for kw in ['def ', 'import torch', '@triton', 'class ']):
                diagram_blocks.append((f, idx, b_clean))

print(f"Found {len(diagram_blocks)} ASCII diagram blocks across {len(files)} files.")

for f, idx, b in diagram_blocks[:6]:
    print(f"\n==================== {f} [Block {idx}] ====================")
    print(b)
