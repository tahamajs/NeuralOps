import sys
import importlib
from pathlib import Path

proj_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(proj_dir))
sys.path.insert(0, str(proj_dir / "Codes"))

# Read canonical_modules.txt and extract clean tokens
modules = []
with open("canonical_modules.txt", "r", encoding="utf-8") as f:
    for line in f:
        clean = line.split("#")[0].strip()
        if clean:
            tok = clean.split()[0].strip()
            if tok:
                modules.append(tok)

if "full" not in modules:
    modules.insert(0, "full")

# Deduplicate preserving order
seen = set()
clean_modules = []
for m in modules:
    if m not in seen:
        seen.add(m)
        clean_modules.append(m)

print(f"Total clean canonical modules: {len(clean_modules)}")

with open("canonical_modules.txt", "w", encoding="utf-8") as f:
    for m in clean_modules:
        f.write(f"{m}\n")

print("Saved clean canonical_modules.txt successfully.")
