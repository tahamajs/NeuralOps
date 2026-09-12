import glob
files = glob.glob('attentions_latex/individual_modules/*.tex')
c = 0
for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        s = fp.read()
    # Check if there is single backslash before [0.3cm]
    if r'\[0.3cm]' in s and r'\\[0.3cm]' not in s:
        s_new = s.replace(r'\[0.3cm]', r'\\[0.3cm]')
        with open(f, 'w', encoding='utf-8') as fp:
            fp.write(s_new)
        c += 1
print(f'Fixed in {c} files.')
