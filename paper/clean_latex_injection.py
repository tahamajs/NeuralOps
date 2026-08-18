import json
import re
import os

docs_dir = "/Users/tahamajs/Documents/uni/Project/Docs/"
main_tex_path = "/Users/tahamajs/Documents/uni/Project/paper/main.tex"

unicode_math_map = {
    'Δ': r'\Delta ', 'Σ': r'\Sigma ', 'α': r'\alpha ', 'ε': r'\epsilon ', 'σ': r'\sigma ', 'τ': r'\tau ',
    '⁸': r'^8', '₁': r'_1', '₂': r'_2', '₃': r'_3', 'ℝ': r'\mathbb{R} ', '⇒': r'\Rightarrow ',
    '∈': r'\in ', '√': r'\sqrt', '≈': r'\approx ', '≤': r'\leq ', '≥': r'\geq ', '≪': r'\ll ',
    '⊗': r'\otimes ', '⊙': r'\odot ', '⌊': r'\lfloor ', '⌋': r'\rfloor ', '²': r'^2', '³': r'^3',
}
unicode_ascii_map = {
    '─': '-', '│': '|', '┌': '+', '┐': '+', '└': '+', '┘': '+', 
    '├': '+', '┤': '+', '┬': '+', '┴': '+', '┼': '+',
    '►': '>', '▼': 'v', '◄': '<', '✅': '[X]', '❌': '[ ]',
    '🚀': '*', '🔥': '*', '🔄': '*', '\u0007': '',
}

def escape_latex_clean(text):
    text = text.replace('\f', '\\f').replace('\t', '\\t').replace('\r', '\\r').replace('\v', '\\v').replace('\b', '\\b')

    for k, v in unicode_ascii_map.items(): text = text.replace(k, v)
    for k, v in unicode_math_map.items(): text = text.replace(k, f"${v}$")
        
    placeholders = {}
    counter = [0]
    
    def add_ph(val, ptype):
        key = f"MPH{counter[0]}E"
        counter[0] += 1
        placeholders[key] = (val, ptype)
        return key

    text = re.sub(r'```(.*?)```', lambda m: add_ph(m.group(1), 'code_block'), text, flags=re.DOTALL)
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: add_ph(m.group(1), 'math_block'), text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.*?)\\\]', lambda m: add_ph(m.group(1), 'math_block'), text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', lambda m: add_ph(m.group(1), 'inline_code'), text)
    text = re.sub(r'\$(.*?)\$', lambda m: add_ph(m.group(1), 'inline_math'), text)

    lines = text.split('\n')
    out_lines = []
    
    list_stack = []
    in_table = False
    
    def close_envs():
        nonlocal in_table
        while list_stack:
            out_lines.append(f'\\end{{{list_stack.pop()[1]}}}')
        if in_table:
            out_lines.append(r'\bottomrule')
            out_lines.append(r'\end{tabular}')
            out_lines.append(r'\end{adjustbox}')
            out_lines.append(r'\end{center}')
            in_table = False

    def escape_line(l):
        l = l.replace('\\', '\\textbackslash{}')
        l = l.replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
        l = l.replace('_', '\\_').replace('{', '\\{').replace('}', '\\}')
        l = l.replace('~', '\\textasciitilde{}').replace('^', '\\textasciicircum{}')
        l = l.replace('$', '\\$')
        
        l = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', l)
        l = re.sub(r'\*(.*?)\*', r'\\textit{\1}', l)
        return l

    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            out_lines.append("")
            continue
            
        match_ph = re.match(r'^MPH\d+E$', stripped)
        if match_ph:
            close_envs()
            val, ptype = placeholders[stripped]
            if ptype == 'code_block':
                val = val.strip()
                val = re.sub(r'^[a-zA-Z0-9]+\n', '', val) 
                out_lines.append(r'\vspace{0.5em}')
                out_lines.append(r'\begin{lstlisting}')
                out_lines.append(val)
                out_lines.append(r'\end{lstlisting}')
                out_lines.append(r'\vspace{0.5em}')
            elif ptype == 'math_block':
                val = val.strip()
                val = re.sub(r'\n\s*\n', '\n', val)
                out_lines.append(r'\begin{center}')
                out_lines.append(r'\begin{adjustbox}{max width=\columnwidth}')
                out_lines.append(r'$\displaystyle')
                out_lines.append(val)
                out_lines.append(r'$')
                out_lines.append(r'\end{adjustbox}')
                out_lines.append(r'\end{center}')
            continue

        if stripped.startswith('#'):
            close_envs()
            level = len(re.match(r'^#+', stripped).group(0))
            title = stripped.lstrip('#').strip()
            title = re.sub(r'^\d+(\.\d+)*\s*', '', title)
            title = escape_line(title)
            
            if level <= 2:
                out_lines.append(f"\\subsubsection*{{{title}}}")
            elif level == 3:
                out_lines.append(f"\\paragraph{{{title}}}")
            else:
                out_lines.append(f"\\subparagraph{{{title}}}")
            continue
            
        list_match = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)', line)
        if list_match:
            if in_table: close_envs()
            indent = len(list_match.group(1))
            marker = list_match.group(2)
            item_text = list_match.group(3)
            
            ltype = 'enumerate' if marker[-1] == '.' else 'itemize'
            
            while list_stack and list_stack[-1][0] > indent:
                out_lines.append(f'\\end{{{list_stack.pop()[1]}}}')
                
            if not list_stack or list_stack[-1][0] < indent:
                list_stack.append((indent, ltype))
                out_lines.append(f'\\begin{{{ltype}}}')
            elif list_stack and list_stack[-1][0] == indent and list_stack[-1][1] != ltype:
                out_lines.append(f'\\end{{{list_stack.pop()[1]}}}')
                list_stack.append((indent, ltype))
                out_lines.append(f'\\begin{{{ltype}}}')
                
            out_lines.append(f"\\item {escape_line(item_text)}")
            continue
            
        if stripped.startswith('|') and stripped.endswith('|'):
            if list_stack: close_envs()
            if re.match(r'^\|[-: \|]+\|$', stripped):
                if in_table:
                    out_lines.append(r'\midrule')
                continue
                
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            if not in_table:
                out_lines.append(r'\begin{center}')
                out_lines.append(r'\begin{adjustbox}{max width=\columnwidth}')
                out_lines.append(r'\renewcommand{\arraystretch}{1.2}')
                cols = 'l' * len(cells)
                out_lines.append(f"\\begin{{tabular}}{{{cols}}}")
                out_lines.append(r'\toprule')
                in_table = True
            
            escaped_cells = [escape_line(c) for c in cells]
            out_lines.append(" & ".join(escaped_cells) + r" \\")
            continue
            
        line_indent = len(line) - len(line.lstrip())
        if list_stack and line_indent == 0:
            while list_stack:
                out_lines.append(f"\\end{{{list_stack.pop()[1]}}}")
                
        if in_table:
            close_envs()
            
        out_lines.append(escape_line(stripped))

    close_envs()
    
    text = '\n'.join(out_lines)
    
    for key, (val, ptype) in placeholders.items():
        if ptype == 'inline_code':
            val = val.replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%', '\\%').replace('#', '\\#').replace('{', '\\{').replace('}', '\\}').replace('&', '\\&').replace('$', '\\$').replace('^', '\\textasciicircum{}').replace('~', '\\textasciitilde{}')
            text = text.replace(key, f"\\texttt{{{val}}}")
        elif ptype == 'inline_math':
            text = text.replace(key, f"${val}$")
            
    return text

with open(os.path.join(docs_dir, "catalog.json"), "r") as f:
    catalog = json.load(f)

module_to_md = {}
for category in ['attentions', 'compressions']:
    for item in catalog.get(category, []):
        name = item['id'].replace('.md', '')
        path = os.path.join(docs_dir, item['path'])
        module_to_md[name] = path
        prefix = re.sub(r'_(attention|compressor)$', '', name)
        module_to_md[prefix] = path

with open(main_tex_path, "r") as f:
    tex_content = f.read()

pattern = re.compile(r'\\subsection\{\\texttt\{([^}]+)\}\}(.*?)(?=\\subsection\{\\texttt\{|\\section|\Z)', re.DOTALL)

def replace_sub(match):
    name = match.group(1).replace(r'\_', '_')
    
    md_path = None
    if name in module_to_md:
        md_path = module_to_md[name]
    elif f"{name}_attention" in module_to_md:
        md_path = module_to_md[f"{name}_attention"]
    elif f"{name}_compressor" in module_to_md:
        md_path = module_to_md[f"{name}_compressor"]
        
    if md_path and os.path.exists(md_path):
        with open(md_path, 'r') as f:
            md_content = f.read()
        latex_content = escape_latex_clean(md_content)
        safe_name = name.replace('_', r'\_')
        return f"\\subsection{{\\texttt{{{safe_name}}}}}\n{latex_content}\n\n"
    else:
        return match.group(0)

new_tex_content = pattern.sub(replace_sub, tex_content)

packages_to_add = r'''
\usepackage{xcolor}
\usepackage{adjustbox}
\usepackage{listings}
\definecolor{codegray}{rgb}{0.96,0.96,0.96}
\definecolor{codeframe}{rgb}{0.75,0.75,0.75}
\lstset{
    backgroundcolor=\color{codegray},
    rulecolor=\color{codeframe},
    frame=single,
    basicstyle=\ttfamily\scriptsize,
    breaklines=true,
    breakatwhitespace=true,
    keepspaces=true,
    showstringspaces=false
}
'''

if r'\definecolor{codegray}' not in new_tex_content:
    new_tex_content = new_tex_content.replace(r'\usepackage{booktabs}', r'\usepackage{booktabs}' + packages_to_add)

with open(main_tex_path, "w") as f:
    f.write(new_tex_content)

print("Fixed lists, enumerate, and styled code blocks.")
