# -*- coding: utf-8 -*-
"""
Fixes unclosed \section, \subsection, \chapter, and \caption commands
in master_thesis.tex and all 100 individual module files.
"""

import os
import glob
import re

BASE_DIR = "/Users/tahamajs/Documents/uni/Project"
THESIS_DIR = os.path.join(BASE_DIR, "thesis_final")
IM_DIR = os.path.join(THESIS_DIR, "attentions_latex", "individual_modules")
MASTER_FILE = os.path.join(THESIS_DIR, "master_thesis.tex")

# 1. Fix master_thesis.tex
with open(MASTER_FILE, "r", encoding="utf-8") as fp:
    lines = fp.readlines()

fixed_master_lines = []
for idx, line in enumerate(lines, 1):
    stripped = line.rstrip()
    if any(stripped.startswith(cmd) for cmd in [r"\section{", r"\subsection{", r"\chapter{", r"\caption{", r"\subsubsection*{"]):
        diff = stripped.count("{") - stripped.count("}")
        if diff > 0:
            stripped += "}" * diff
            print(f"Fixed master line {idx}: {stripped}")
    fixed_master_lines.append(stripped + "\n")

with open(MASTER_FILE, "w", encoding="utf-8") as fp:
    fp.writelines(fixed_master_lines)

# 2. Fix all 100 module files
with open(os.path.join(THESIS_DIR, "chapter_attentions.tex"), "r", encoding="utf-8") as fp:
    ch_lines = fp.readlines()

target_mods = [os.path.basename(l.split("{")[1].split("}")[0]) for l in ch_lines if r"\input{" in l]

fixed_modules = 0
for m in target_mods:
    fpath = os.path.join(IM_DIR, m)
    if not os.path.exists(fpath):
        continue

    with open(fpath, "r", encoding="utf-8") as fp:
        m_lines = fp.readlines()

    fixed_m_lines = []
    file_changed = False
    for idx, line in enumerate(m_lines, 1):
        stripped = line.rstrip()
        if any(stripped.startswith(cmd) for cmd in [r"\section{", r"\subsection{", r"\subsubsection*{", r"\caption{"]):
            diff = stripped.count("{") - stripped.count("}")
            if diff > 0:
                stripped += "}" * diff
                file_changed = True
        fixed_m_lines.append(stripped + "\n")

    if file_changed:
        with open(fpath, "w", encoding="utf-8") as fp:
            fp.writelines(fixed_m_lines)
        fixed_modules += 1

print(f"Fixed unclosed section/subsection headers in {fixed_modules} module files.")
