with open('master_thesis.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Remove mdframed
text = text.replace(r'\usepackage[framemethod=TikZ]{mdframed}', '')

# 2. Fix fonts to use exact disk files with clean bold and no outline
old_font = r"""\settextfont[BoldFont={X Nazanin Bold}, ItalicFont={X Nazanin Italic}, BoldItalicFont={X Nazanin Bold Italic}]{X Nazanin}
\setdigitfont[BoldFont={X Nazanin Bold}]{X Nazanin}"""

new_font = r"""\settextfont[
  Path = /Users/tahamajs/Library/Fonts/,
  BoldFont = {X Nazanin Bold.ttf},
  ItalicFont = {X Nazanin Italic.ttf}
]{X Nazanin.ttf}
\setdigitfont[
  Path = /Users/tahamajs/Library/Fonts/,
  BoldFont = {X Nazanin Bold.ttf}
]{X Nazanin.ttf}"""

text = text.replace(old_font, new_font)

# 3. Fix unescaped PASS_LOW in caption
text = text.replace(r'\caption{توزیع وضعیت‌های \lr{PASS}، \lr{PASS_LOW} و \lr{FAIL}',
                    r'\caption{توزیع وضعیت‌های \lr{PASS}، \lr{PASS\_LOW} و \lr{FAIL}')

# 4. Update title and headings to 100 top modules
text = text.replace('بیش از ۱۰۰ مکانیزم', '۱۰۰ مکانیزم برتر')
text = text.replace('۱۳۶ معماری توجه', '۱۰۰ معماری توجه برتر')
text = text.replace('100+ Attention Mechanisms', '100 Top Attention Mechanisms')

with open('master_thesis.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("Updated master_thesis.tex successfully!")
