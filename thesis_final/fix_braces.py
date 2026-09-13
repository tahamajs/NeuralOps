import glob
files = glob.glob('attentions_latex/individual_modules/*.tex')
c = 0
for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        s = fp.read()
    if '}}}\n' in s:
        s_new = s.replace('}}}\n', '}}\n')
        with open(f, 'w', encoding='utf-8') as fp:
            fp.write(s_new)
        c += 1
print(f'Fixed extra braces in {c} files.')
