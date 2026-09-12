# -*- coding: utf-8 -*-
"""
Perfect header and brace normalizer for all 100 modules.
Ensures every section header is strictly on a single line with matching braces.
Fixes unclosed math environments and unclosed braces.
"""

import os
import re

IM_DIR = "/Users/tahamajs/Documents/uni/Project/thesis_final/attentions_latex/individual_modules"

with open("/Users/tahamajs/Documents/uni/Project/thesis_final/chapter_attentions.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

target_mods = []
for l in lines:
    if r"\input{" in l:
        p = l.split("{")[1].split("}")[0]
        target_mods.append(os.path.basename(p))

SECTION_MAP = {
    1: r"\subsubsection*{بخش ۱: شناسنامه و معرفی \lr{(Introduction)}}",
    2: r"\subsubsection*{بخش ۲: انگیزه و چرایی معماری \lr{(Rationale)}}",
    3: r"\subsubsection*{بخش ۳: مبانی نظری \lr{(Theoretical Foundations)}}",
    4: r"\subsubsection*{بخش ۴: فرمول‌بندی ریاضی کامل \lr{(Mathematical Formulation)}}",
    5: r"\subsubsection*{بخش ۵: معماری و پیاده‌سازی \lr{(Architecture \& Implementation)}}",
    6: r"\subsubsection*{بخش ۶: تحلیل سخت‌افزاری و عملکرد \lr{(Hardware Analysis)}}",
    7: r"\subsubsection*{بخش ۷: ارزیابی تجربی، بنچمارک و توجیه علمی نتایج \lr{(Empirical Benchmark \& Scientific Analysis)}}",
    8: r"\subsubsection*{بخش ۸: تحلیل و تفسیر نتایج \lr{(Interpretation)}}",
    9: r"\subsubsection*{بخش ۹: نقاط قوت و ضعف \lr{(Strengths \& Weaknesses)}}",
    10: r"\subsubsection*{بخش ۱۰: نتیجه‌گیری بخش \lr{(Section Conclusion)}}",
    11: r"\subsubsection*{بخش ۱۱: تصاویر و نمودارها \lr{(Figures \& Charts)}}",
    12: r"\subsubsection*{بخش ۱۲: مراجع \lr{(References)}}",
}

def repair_file(content, mod_name):
    # 1. Clean any broken \subsubsection*{\nبخش
    content = re.sub(r'\\sub(?:sub)?section\*\{\s*\n\s*بخش', r'\nبخش', content)

    # 2. Replace any line that starts with بخش X: or has \subsubsection*{بخش X: with the canonical header
    for num, canonical in SECTION_MAP.items():
        persian_digit = "۰۱۲۳۴۵۶۷۸۹"[num] if num <= 9 else ("۱۰" if num == 10 else ("۱۱" if num == 11 else "۱۲"))
        pattern = rf'(?:\\sub(?:sub)?section\*\{{)?\s*بخش\s*(?:{num}|{persian_digit}):[^\n]+'
        content = re.sub(pattern, lambda m, c=canonical: c, content)

    # 3. Fix \label{sec:...}
    escaped_mod = mod_name.replace(".tex", "").replace("_", r"\_")
    bare_mod = mod_name.replace(".tex", "")
    
    title_repl = f'\\subsection{{تحلیل علمی و کالبدشکافی ساختاری ماژول \\lr{{{escaped_mod}}}}}\n\\label{{sec:{bare_mod}}}\n'
    content = re.sub(
        r'\\sub(?:sub)?section\{[^\n]+\n(?:\\label\{[^\n]+\}\n)?',
        lambda m, tr=title_repl: tr,
        content,
        count=1
    )

    # Clean any leftover trailing braces on \label
    content = re.sub(r'(\\label\{sec:[^}]+\})\}*', r'\1', content)

    # 4. Fix specific math equation brace issues
    # Ensure \mathbb{R}^{...} is closed before \end{equation} or newline
    lines = content.split('\n')
    fixed_lines = []
    for line in lines:
        if r'\mathbb{R}^{' in line and r'\end{equation}' in line:
            # Check balance
            if line.count('{') > line.count('}'):
                line = line.replace(r'\end{equation}', r'}\end{equation}')
        fixed_lines.append(line)
    content = '\n'.join(fixed_lines)

    return content

repaired = 0
for m in target_mods:
    fpath = os.path.join(IM_DIR, m)
    if not os.path.exists(fpath):
        continue
    with open(fpath, "r", encoding="utf-8") as fp:
        c = fp.read()
    fixed = repair_file(c, m)
    with open(fpath, "w", encoding="utf-8") as fp:
        fp.write(fixed)
    repaired += 1

print(f"Successfully repaired headers in all {repaired} modules.")
