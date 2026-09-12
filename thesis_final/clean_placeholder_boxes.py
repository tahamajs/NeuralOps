import os, glob

all_tex = glob.glob('attentions_latex/individual_modules/*.tex')
cleaned_count = 0

for fpath in all_tex:
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()

    changed = False
    while True:
        pos = text.find(r'\framebox[0.9\textwidth]')
        if pos == -1:
            pos = text.find(r'(بر اساس پرامپت فنی فوق')
        if pos == -1:
            break

        fig_start = text.rfind(r'\begin{figure}', 0, pos)
        fig_end = text.find(r'\end{figure}', pos)
        if fig_start != -1 and fig_end != -1:
            fig_end += len(r'\end{figure}')
            # Also check if followed by optional vspace / hrule
            tail = text[fig_end:fig_end+200]
            extra = 0
            for prefix in ['\n\n\\vspace{0.5cm}\n\\hrule\n\\vspace{0.5cm}', '\n\\vspace{0.5cm}\n\\hrule\n\\vspace{0.5cm}']:
                if tail.startswith(prefix):
                    extra = len(prefix)
                    break
            text = text[:fig_start] + text[fig_end+extra:]
            changed = True
        else:
            break

    if changed:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(text)
        cleaned_count += 1

print(f'Successfully removed placeholder boxes from {cleaned_count} files.')
