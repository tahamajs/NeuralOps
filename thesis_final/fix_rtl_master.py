with open('master_thesis.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Remove titlesec which forces LTR / left-aligned headings
text = text.replace(r'\usepackage{titlesec}', '')

# 2. Add \setRTL right after \begin{document}
if r'\setRTL' not in text:
    text = text.replace(r'\begin{document}', "\\begin{document}\n\\setRTL")

with open('master_thesis.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated master_thesis.tex: removed titlesec and enforced \setRTL.")
