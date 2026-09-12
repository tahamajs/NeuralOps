# -*- coding: utf-8 -*-
import os
import glob
import re

IM_DIR = "/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules"

with open("/Users/tahamajs/Documents/uni/Project/thesis_final/chapter_attentions.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

target_mods = []
for l in lines:
    if r"\input{" in l:
        p = l.split("{")[1].split("}")[0]
        target_mods.append(os.path.basename(p))

print(f"Checking {len(target_mods)} modules...")

errors = []
for m in target_mods:
    fpath = os.path.join(IM_DIR, m)
    if not os.path.exists(fpath):
        continue
    with open(fpath, "r", encoding="utf-8") as fp:
        lines = fp.readlines()

    # Track line by line brace balance
    open_curly = 0
    in_equation = False
    eq_start = 0

    for idx, line in enumerate(lines, 1):
        # Ignore comments
        stripped = line.split('%')[0]

        # Check section header line brace balance specifically
        if r'\subsubsection*{' in stripped or r'\subsection{' in stripped:
            b_open = stripped.count('{')
            b_close = stripped.count('}')
            if b_open != b_close:
                errors.append((m, idx, f"Mismatched braces in header: open={b_open}, close={b_close}: {line.strip()}"))

        # Check equation environment
        if r'\begin{equation}' in stripped:
            if in_equation:
                errors.append((m, idx, f"Nested begin equation (started line {eq_start})"))
            in_equation = True
            eq_start = idx
        if r'\end{equation}' in stripped:
            if not in_equation:
                errors.append((m, idx, "Unmatched end equation"))
            in_equation = False

        # Count total braces in file
        open_curly += stripped.count('{') - stripped.count('}')

    if open_curly != 0:
        errors.append((m, len(lines), f"Total brace mismatch in file: net open braces = {open_curly}"))

print(f"Total errors found: {len(errors)}")
for err in errors[:25]:
    print(" ", err)
