import re
from pathlib import Path

repo_root = Path("/mnt/d/SVD_linear_Attention")

# 1. table_all_131_attentions.tex
p1 = repo_root / "paper/iclr2027-submission/tables/table_all_131_attentions.tex"
with open(p1, "r", encoding="utf-8") as f:
    text1 = f.read()

text1 = text1.replace("All 131 Canonical Attention Mechanisms", "All 130 Canonical Attention Mechanisms")
lines1 = text1.splitlines()
new1 = []
idx1 = 1
for line in lines1:
    if "adasplash" in line and "\\texttt{" in line:
        continue
    # Match lines like "4 & \texttt{..." or "131 & \texttt{..."
    m = re.match(r"^(\d+)\s*(&\s*\\texttt\{.*)$", line)
    if m:
        line = f"{idx1} {m.group(2)}"
        idx1 += 1
    new1.append(line)

with open(p1, "w", encoding="utf-8") as f:
    f.write("\n".join(new1) + "\n")
print("table_all_131_attentions.tex: count =", idx1 - 1)

# 2. chapter_all_modules_appendix.tex
p2 = repo_root / "thesis_final/chapter_all_modules_appendix.tex"
with open(p2, "r", encoding="utf-8") as f:
    text2 = f.read()

text2 = text2.replace("۱۳۱ مکانیزم متعارف توجه", "۱۳۰ مکانیزم متعارف توجه")
lines2 = text2.splitlines()
new2 = []
idx2 = 1
in_table = False
for line in lines2:
    if "\\begin{longtable}" in line:
        in_table = True
    if in_table and "\\end{longtable}" in line:
        in_table = False

    if in_table and "adasplash" in line and "\\texttt{" in line:
        continue

    if in_table:
        m = re.match(r"^(\d+)\s*(&\s*\\texttt\{.*)$", line)
        if m:
            line = f"{idx2} {m.group(2)}"
            idx2 += 1
    new2.append(line)

with open(p2, "w", encoding="utf-8") as f:
    f.write("\n".join(new2) + "\n")
print("chapter_all_modules_appendix.tex: count =", idx2 - 1)

# 3. app_131_attentions_empirical.tex
p3 = repo_root / "thesis_final/app_131_attentions_empirical.tex"
with open(p3, "r", encoding="utf-8") as f:
    text3 = f.read()

lines3 = text3.splitlines()
new3 = []
idx3 = 1
for line in lines3:
    if "adasplash" in line and "\\texttt{" in line:
        continue
    m = re.match(r"^(\d+)\s*(&\s*\\texttt\{.*)$", line)
    if m:
        line = f"{idx3} {m.group(2)}"
        idx3 += 1
    new3.append(line)

with open(p3, "w", encoding="utf-8") as f:
    f.write("\n".join(new3) + "\n")
print("app_131_attentions_empirical.tex: count =", idx3 - 1)
