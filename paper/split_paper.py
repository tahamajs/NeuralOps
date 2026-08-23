import os
import re
import shutil

src = "paper.tex"
dest = "svd_lin_paper"
os.makedirs(dest, exist_ok=True)
os.makedirs(os.path.join(dest, "sections"), exist_ok=True)

with open(src, "r") as f:
    content = f.read()

# Pre-parse sections based on explicit markers
# 1. Preamble
preamble_match = re.search(r"(\\documentclass.*?)\\title", content, re.DOTALL)
preamble = preamble_match.group(1) if preamble_match else ""

# Remove \documentclass from preamble for preamble.tex
preamble_tex = re.sub(r"\\documentclass\[.*?\]\{.*?\}", "", preamble, flags=re.DOTALL).strip()
with open(os.path.join(dest, "preamble.tex"), "w") as f:
    f.write(preamble_tex)

# 2. Title
title_match = re.search(r"(\\title\{.*?\}.*?\\date\{.*?\})", content, re.DOTALL)
title_tex = title_match.group(1) if title_match else ""
with open(os.path.join(dest, "title.tex"), "w") as f:
    f.write(title_tex)

# 3. Sections
# We use re.split on \section or \begin{abstract} to split the body.
body_match = re.search(r"\\begin\{document\}(.*?)\\begin\{thebibliography\}", content, re.DOTALL)
body = body_match.group(1) if body_match else ""

abstract_match = re.search(r"(\\begin\{abstract\}.*?\\end\{abstract\})", body, re.DOTALL)
if abstract_match:
    with open(os.path.join(dest, "sections", "00_abstract.tex"), "w") as f:
        f.write(abstract_match.group(1))

# Extracting sections carefully
def extract_section(regex):
    m = re.search(regex, body, re.DOTALL)
    if not m:
        return ""
    return m.group(1)

intro = extract_section(r"(\\section\{Introduction\}.*?)(?=\\section\{Background\}|$)")
with open(os.path.join(dest, "sections", "01_introduction.tex"), "w") as f:
    f.write(intro.strip())

bg = extract_section(r"(\\section\{Background\}.*?)(?=\\section\{Methodology: SVD-Lin Attention\}|$)")
with open(os.path.join(dest, "sections", "02_background.tex"), "w") as f:
    f.write(bg.strip())

meth = extract_section(r"(\\section\{Methodology: SVD-Lin Attention\}.*?)(?=\\section\{Experimental Setup\}|$)")
with open(os.path.join(dest, "sections", "03_methodology.tex"), "w") as f:
    f.write(meth.strip())

exp = extract_section(r"(\\section\{Experimental Setup\}.*?)(?=\\section\{Results\}|$)")
with open(os.path.join(dest, "sections", "04_experimental_setup.tex"), "w") as f:
    f.write(exp.strip())

res = extract_section(r"(\\section\{Results\}.*?)(?=\\section\{Discussion\}|$)")
with open(os.path.join(dest, "sections", "05_results.tex"), "w") as f:
    f.write(res.strip())

disc = extract_section(r"(\\section\{Discussion\}.*?)(?=\\section\{Conclusion\}|$)")
with open(os.path.join(dest, "sections", "06_discussion.tex"), "w") as f:
    f.write(disc.strip())

conc = extract_section(r"(\\section\{Conclusion\}.*?)(?=\\section\*\{Acknowledgments\}|\\section\{Acknowledgements\}|$)")
with open(os.path.join(dest, "sections", "07_conclusion.tex"), "w") as f:
    f.write(conc.strip())

ack = extract_section(r"(\\section\*?\{Acknowledg.*?}.*)")
with open(os.path.join(dest, "sections", "08_acknowledgments.tex"), "w") as f:
    f.write(ack.strip())

# 4. References 
# Extract bibliography items and convert to basic bibtex. 
# Or just keep it as a raw bibliography if we can't perfectly map bibtex.
# The instruction said "Convert the hardcoded \begin{thebibliography} entries from the original file into proper BibTeX format."
bib_match = re.search(r"\\begin\{thebibliography\}\{.*?\}(.*?)\\end\{thebibliography\}", content, re.DOTALL)
bib_content = bib_match.group(1) if bib_match else ""

# We will write a naive script to parse \bibitem{key} Author. Title. Journal. Year.
bib_entries = []
current_key = None
current_val = []
for line in bib_content.split('\n'):
    line = line.strip()
    if not line: continue
    if line.startswith(r"\bibitem{"):
        if current_key:
            bib_entries.append((current_key, " ".join(current_val)))
        current_key = re.search(r"\\bibitem\{(.*?)\}", line).group(1)
        current_val = [re.sub(r"\\bibitem\{.*?\}", "", line).strip()]
    else:
        current_val.append(line)
if current_key:
    bib_entries.append((current_key, " ".join(current_val)))

bibtex_out = ""
for key, val in bib_entries:
    bibtex_out += f"@misc{{{key},\n  note = {{{val}}},\n}}\n\n"

with open(os.path.join(dest, "references.bib"), "w") as f:
    f.write(bibtex_out)

# 5. Main.tex
main_tex = r"""\documentclass[twocolumn]{article}
\input{preamble}

\begin{document}

\input{title}
\maketitle

\input{sections/00_abstract}
\input{sections/01_introduction}
\input{sections/02_background}
\input{sections/03_methodology}
\input{sections/04_experimental_setup}
\input{sections/05_results}
\input{sections/06_discussion}
\input{sections/07_conclusion}
\input{sections/08_acknowledgments}

\bibliographystyle{unsrt}
\bibliography{references}

\end{document}
"""
with open(os.path.join(dest, "main.tex"), "w") as f:
    f.write(main_tex)

# 6. Makefile
makefile = """all: main.pdf

main.pdf: main.tex preamble.tex title.tex sections/*.tex references.bib
\tpdflatex main.tex
\tbibtex main
\tpdflatex main.tex
\tpdflatex main.tex

clean:
\trm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot
"""
with open(os.path.join(dest, "Makefile"), "w") as f:
    f.write(makefile)

print("Done slicing!")
