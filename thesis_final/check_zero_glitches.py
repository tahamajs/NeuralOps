import glob
import os
import re

files = glob.glob("thesis_final/attentions_latex/individual_modules/*.tex")

zero_glitches = []

for f in files:
    with open(f, "r", encoding="utf-8") as fp:
        c = fp.read()
    
    issues = []
    if "-100.0%" in c or "-100%" in c:
        issues.append("-100% memory reduction")
    if "0.00x" in c or "0.0x" in c:
        issues.append("0.00x throughput multiplier")
    if "سطح \\lr{0.00\\%}" in c or "سطح \\lr{0.0\\%}" in c:
        issues.append("F1-Score dropped to 0.00%")

    if issues:
        zero_glitches.append((os.path.basename(f), issues))

print(f"Total files with boilerplate zero/100 glitches: {len(zero_glitches)}")
for fname, iss in zero_glitches[:15]:
    print(f" - {fname:30}: {iss}")
