import os, glob

with open('thesis_final/chapter_attentions.tex') as f:
    modules = [l.strip().replace(r'\input{', '').rstrip('}') for l in f if 'input{' in l]

for m in modules:
    if not os.path.exists(m):
        continue
    with open(m, 'r', encoding='utf-8') as fl:
        lines = fl.readlines()
    
    modified = False
    new_lines = []
    for l in lines:
        # Fix caption missing brace
        if r'\caption{' in l and l.count('{') > l.count('}'):
            l = l.rstrip() + "}\n"
            modified = True
        # Fix \textbf{مدل پایه \lr{(Baseline)}
        if r'\textbf{مدل پایه \lr{(Baseline)}' in l and r'\textbf{مدل پایه \lr{(Baseline)}}' not in l:
            l = l.replace(r'\textbf{مدل پایه \lr{(Baseline)}', r'\textbf{مدل پایه \lr{(Baseline)}}')
            modified = True
        new_lines.append(l)
    
    if modified:
        with open(m, 'w', encoding='utf-8') as fl:
            fl.writelines(new_lines)
        print(f"Fixed table braces in {os.path.basename(m)}")

