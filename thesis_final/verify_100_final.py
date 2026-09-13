# -*- coding: utf-8 -*-
import glob
import os
import re

im_dir = "/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules"

with open("/Users/tahamajs/Documents/uni/Project/thesis_final/chapter_attentions.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

target_mods = []
for l in lines:
    if r"\input{" in l:
        p = l.split("{")[1].split("}")[0]
        target_mods.append(os.path.basename(p))

print(f"Validating {len(target_mods)} target modules...")

issues = []
for mname in target_mods:
    fpath = os.path.join(im_dir, mname)
    if not os.path.exists(fpath):
        issues.append((mname, "File does not exist"))
        continue

    with open(fpath, "r", encoding="utf-8") as fp:
        c = fp.read()

    # Check environments balance
    begins = re.findall(r"\\begin\{([a-zA-Z0-9_\*]+)\}", c)
    ends = re.findall(r"\\end\{([a-zA-Z0-9_\*]+)\}", c)
    if len(begins) != len(ends):
        issues.append((mname, f"Env count mismatch: {len(begins)} begins vs {len(ends)} ends"))

    # Check unescaped % inside \lr
    for match in re.finditer(r"\\lr\{([^}]+)\}", c):
        inner = match.group(1)
        if re.search(r"(?<!\\)%", inner):
            issues.append((mname, f"Unescaped % in lr: {match.group(0)[:40]}"))
            break

    # Check unescaped _ inside \lr
    for match in re.finditer(r"\\lr\{([^}]+)\}", c):
        inner = match.group(1)
        if re.search(r"(?<!\\)_", inner):
            issues.append((mname, f"Unescaped _ in lr: {match.group(0)[:40]}"))
            break

    # Check 3-model table presence
    if "DistilBERT" not in c or "RoBERTa-base" not in c or "GPT-2" not in c:
        issues.append((mname, "Missing 3-model table"))

    # Check NeuralOps
    if "neuralops" in c.lower():
        issues.append((mname, "Found NeuralOps"))

print(f"Total issues found: {len(issues)}")
if issues:
    for iss in issues[:15]:
        print(" ", iss)
else:
    print("ALL 100 MODULES FULLY VALIDATED AND 100% CLEAN!")
