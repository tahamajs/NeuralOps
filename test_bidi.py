import subprocess, os

test_tex = r'''\documentclass{article}
\usepackage{xepersian}
\settextfont{X Nazanin}
\setlatintextfont{Times New Roman}
\begin{document}
روش اول: \lr{(SVD)}

روش دوم: (\lr{SVD})

جدول اول: دقت \lr{(Accuracy)} و شاخص \lr{(F1-Score)}

جدول دوم: دقت (\lr{Accuracy}) و شاخص (\lr{F1-Score})
\end{document}
'''
with open('test_bidi.tex', 'w', encoding='utf-8') as f:
    f.write(test_tex)

subprocess.run(['xelatex', '-interaction=nonstopmode', 'test_bidi.tex'], stdout=subprocess.DEVNULL)

res = subprocess.run(['pdftotext', 'test_bidi.pdf', '-'], stdout=subprocess.PIPE, text=True)
print("EXTRACTED TEXT:")
print(res.stdout)
