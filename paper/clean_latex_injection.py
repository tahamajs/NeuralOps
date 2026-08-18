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
    # Fix literal control characters caused by bad string generation previously!
    text = text.replace('\f', '\\f')
    text = text.replace('\t', '\\t')
    text = text.replace('\r', '\\r')
    text = text.replace('\v', '\\v')
    text = text.replace('\b', '\\b')

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
    
    in_list = False
    in_table = False
    
    def close_envs():
        nonlocal in_list, in_table
        if in_list:
            out_lines.append(r'\end{itemize}')
            in_list = False
        if in_table:
            out_lines.append(r'\bottomrule')
            out_lines.append(r'\end{tabular}')
            out_lines.append(r'\end{center}')
            in_table = False

    def escape_line(l):
        # Escape special chars
        l = l.replace('\\', '\\textbackslash{}')
        l = l.replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
        l = l.replace('_', '\\_').replace('{', '\\{').replace('}', '\\}')
        l = l.replace('~', '\\textasciitilde{}').replace('^', '\\textasciicircum{}')
        l = l.replace('$', '\\$')
        
        # bold / italic
        l = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', l)
        l = re.sub(r'\*(.*?)\*', r'\\textit{\1}', l)
        return l

    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            close_envs()
            out_lines.append("")
            continue
            
        # Is it a block placeholder?
        match_ph = re.match(r'^MPH\d+E$', stripped)
        if match_ph:
            close_envs()
            val, ptype = placeholders[stripped]
            if ptype == 'code_block':
                val = val.strip()
                val = re.sub(r'^[a-zA-Z0-9]+\n', '', val) # remove lang
                out_lines.append(r'\vspace{0.5em}')
                out_lines.append(r'\begin{verbatim}')
                out_lines.append(val)
                out_lines.append(r'\end{verbatim}')
                out_lines.append(r'\vspace{0.5em}')
            elif ptype == 'math_block':
                val = val.strip()
                val = re.sub(r'\n\s*\n', '\n', val)
                out_lines.append(r'\begin{equation*}')
                out_lines.append(val)
                out_lines.append(r'\end{equation*}')
            continue

        # Headers
        if stripped.startswith('#'):
            close_envs()
            level = len(re.match(r'^#+', stripped).group(0))
            title = stripped.lstrip('#').strip()
            title = re.sub(r'^\d+(\.\d+)*\s*', '', title) # remove numbering like "2.1 "
            title = escape_line(title)
            
            if level <= 2:
                out_lines.append(f"\\subsubsection*{{{title}}}")
            elif level == 3:
                out_lines.append(f"\\paragraph{{{title}}}")
            else:
                out_lines.append(f"\\subparagraph{{{title}}}")
            continue
            
        # Lists
        if re.match(r'^[-*]\s+', stripped):
            if in_table: close_envs()
            if not in_list:
                out_lines.append(r'\begin{itemize}')
                in_list = True
            item_text = re.sub(r'^[-*]\s+', '', line).strip()
            out_lines.append(f"\\item {escape_line(item_text)}")
            continue
            
        # Tables
        if stripped.startswith('|') and stripped.endswith('|'):
            if in_list: close_envs()
            # If it's a separator line like |---|---| skip it but we know we are in table
            if re.match(r'^\|[-: \|]+\|$', stripped):
                if in_table:
                    out_lines.append(r'\midrule')
                continue
                
            cells = [c.strip() for c in stripped.strip('|').split('|')]
            if not in_table:
                # Start table without float!
                out_lines.append(r'\begin{center}')
                out_lines.append(r'\renewcommand{\arraystretch}{1.2}')
                cols = 'l' * len(cells)
                out_lines.append(f"\\begin{{tabular}}{{{cols}}}")
                out_lines.append(r'\toprule')
                in_table = True
            
            # Escape cells
            escaped_cells = [escape_line(c) for c in cells]
            out_lines.append(" & ".join(escaped_cells) + r" \\")
            continue
            
        # Normal text
        if in_list and not line.startswith('  '):
            close_envs()
        elif in_table:
            close_envs()
            
        out_lines.append(escape_line(line))

    close_envs()
    
    text = '\n'.join(out_lines)
    
    # Restore inline placeholders
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
    name = match.group(1)
    
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
        return f"\\subsection{{\\texttt{{{name}}}}}\n{latex_content}\n\n"
    else:
        return match.group(0)

new_tex_content = pattern.sub(replace_sub, tex_content)

with open(main_tex_path, "w") as f:
    f.write(new_tex_content)

print("Float-free beautiful conversion complete with control-character unescaping.")
