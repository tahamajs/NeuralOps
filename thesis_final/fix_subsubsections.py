import glob
import os
import re

files = glob.glob("thesis_final/attentions_latex/individual_modules/*.tex")
fixed_count = 0

for f in files:
    with open(f, "r", encoding="utf-8") as fp:
        content = fp.read()

    orig = content
    # If line starts with "بخش ۱: شناسنامه" and ends with "}" without \subsubsection*{
    content = re.sub(
        r"(?m)^بخش ۱: شناسنامه و معرفی \\lr\{\(Introduction\)\}\}",
        r"\\subsubsection*{بخش ۱: شناسنامه و معرفی \\lr{(Introduction)}}",
        content
    )
    content = re.sub(
        r"(?m)^بخش ۱:([^\n]+)\}\s*$",
        r"\\subsubsection*{بخش ۱:\1}",
        content
    )

    if content != orig:
        with open(f, "w", encoding="utf-8") as fp:
            fp.write(content)
        fixed_count += 1

print(f"Fixed missing \\subsubsection* prefix in {fixed_count} files.")
