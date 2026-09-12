import glob
import os
import re

files = glob.glob("thesis_final/attentions_latex/individual_modules/*.tex")
fixed_count = 0

for f in files:
    with open(f, "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    modified = False
    new_lines = []
    for l in lines:
        # If line contains \lr{...%} where % is not preceded by backslash
        if "&" in l and re.search(r"\\lr\{[^}]*(?<!\\)%\}?", l):
            # Replace (?<!\\)%} with \%}
            l = re.sub(r"(?<!\\)%\}?", r"\\%}", l)
            modified = True
        new_lines.append(l)

    if modified:
        with open(f, "w", encoding="utf-8") as fp:
            fp.writelines(new_lines)
        fixed_count += 1

print(f"Fixed unescaped percent in {fixed_count} files.")
