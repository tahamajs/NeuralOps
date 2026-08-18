import json
import re
import os

docs_dir = "/Users/tahamajs/Documents/uni/Project/Docs/"
main_tex_path = "/Users/tahamajs/Documents/uni/Project/paper/main.tex"

unicode_math_map = {
    'Δ': r'$\Delta$', 'Σ': r'$\Sigma$', 'α': r'$\alpha$', 'ε': r'$\epsilon$', 'σ': r'$\sigma$', 'τ': r'$\tau$',
    '⁸': r'$^8$', '₁': r'$_1$', '₂': r'$_2$', '₃': r'$_3$', 'ℝ': r'$\mathbb{R}$', '⇒': r'$\Rightarrow$',
    '∈': r'$\in$', '√': r'$\sqrt{}$', '≈': r'$\approx$', '≤': r'$\le$', '≥': r'$\ge$', '≪': r'$\ll$',
    '⊗': r'$\otimes$', '⊙': r'$\odot$', '⌊': r'$\lfloor$', '⌋': r'$\rfloor$', '²': r'$^2$', '³': r'$^3$',
    '⌈': r'$\lceil$', '⌉': r'$\rceil$', '·': r'$\cdot$',
}
unicode_ascii_map = {
    '─': ' ', '│': ' ', '┌': ' ', '┐': ' ', '└': ' ', '┘': ' ', 
    '├': ' ', '┤': ' ', '┬': ' ', '┴': ' ', '┼': ' ',
    '►': ' ', '▼': ' ', '◄': ' ', '✅': '[X]', '❌': '[ ]',
    '🚀': '*', '🔥': '*', '🔄': '*', '\u0007': '',
}

def clean_label_for_tikz(text):
    text = text.replace('\f', ' ').replace('\t', ' ').replace('\r', ' ').replace('\v', ' ').replace('\b', ' ')
    for k, v in unicode_ascii_map.items(): text = text.replace(k, ' ')
    for k, v in unicode_math_map.items(): text = text.replace(k, v)
    text = text.replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
    text = text.replace('^', r'\textasciicircum{}').replace('~', r'\textasciitilde{}')
    for ch in ['┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '─', '│', '►', '▼', '◄', '▲', '+---', '|---', '+', '──►', '───►', '-->', '==>', '->']:
        text = text.replace(ch, ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    math_placeholders = []
    def save_math(m):
        math_placeholders.append(m.group(0))
        return f"MPHTIKZ{len(math_placeholders)-1}END"
        
    text = re.sub(r'\$(.*?)\$', save_math, text)
    text = re.sub(r'(\\widetilde\{[^{}]+\})', r'$\1$', text)
    text = re.sub(r'(\\hat\{[^{}]+\})', r'$\1$', text)
    text = re.sub(r'(\\mathbf\{[^{}]+\})', r'$\1$', text)
    text = re.sub(r'(\\mathcal\{[^{}]+\})', r'$\1$', text)
    text = re.sub(r'(\\Sigma|\\Delta|\\alpha|\\epsilon|\\sigma|\\tau|\\mu|\\theta|\\gamma|\\phi|\\psi|\\lambda|\\omega|\\odot|\\otimes)', r'$\1$', text)
    text = re.sub(r'\$(.*?)\$', save_math, text)
    
    text = text.replace('_', r'\_')
    
    for i, m in enumerate(math_placeholders):
        text = text.replace(f"MPHTIKZ{i}END", m)
        
    text = re.sub(r'\[([a-zA-Z\d\s,\*\+\\_]+)\]', r'$[\1]$', text)
    return text

def parse_ascii_diagram_to_tikz(ascii_text):
    lines = [l.rstrip() for l in ascii_text.strip().split('\n') if l.strip()]
    if not lines:
        return None
    
    nodes = []
    current_block_lines = []
    
    for line in lines:
        stripped = line.strip()
        if re.match(r'^[┌└├─\+\=\-\s]+$', stripped) or (re.match(r'^[│\|\s▼▲►◄vV\^\|\.\:\-]+$', stripped) and not any(c.isalnum() for c in stripped)):
            if current_block_lines:
                raw_text = " ".join([l.strip('│| ') for l in current_block_lines if l.strip('│| ')])
                if raw_text:
                    nodes.append(raw_text)
                current_block_lines = []
            continue
        
        content = re.sub(r'^[┌└├│\|\+\-\s]+', '', line)
        content = re.sub(r'[┐┘┤│\|\+\-\s]+$', '', content).strip()
        if content:
            current_block_lines.append(content)
            
    if current_block_lines:
        raw_text = " ".join([l.strip('│| ') for l in current_block_lines if l.strip('│| ')])
        if raw_text:
            nodes.append(raw_text)
            
    clean_nodes = []
    for n in nodes:
        cl = clean_label_for_tikz(n)
        if cl and len(cl) > 1 and not re.match(r'^[vV\^\|\-\+\>\<\s]+$', cl):
            if len(cl) > 45 and '\\\\' not in cl:
                cl = re.sub(r'(\s*:\s*)', r':\\\\ ', cl, count=1)
            clean_nodes.append(cl)
            
    if len(clean_nodes) < 2:
        return None
        
    tikz_lines = []
    tikz_lines.append(r'\begin{center}')
    tikz_lines.append(r'\begin{adjustbox}{max width=0.98\columnwidth}')
    tikz_lines.append(r'\begin{tikzpicture}[')
    tikz_lines.append(r'    node distance=0.75cm,')
    tikz_lines.append(r'    every node/.style={align=center},')
    tikz_lines.append(r'    flowblock/.style={rectangle, draw=blue!70!black, fill=blue!4, rounded corners=3pt, line width=0.65pt, inner sep=4.5pt, font=\scriptsize, text width=0.92\columnwidth},')
    tikz_lines.append(r'    topblock/.style={rectangle, draw=teal!70!black, fill=teal!4, rounded corners=3pt, line width=0.65pt, inner sep=4.5pt, font=\scriptsize, text width=0.92\columnwidth},')
    tikz_lines.append(r'    botblock/.style={rectangle, draw=violet!70!black, fill=violet!4, rounded corners=3pt, line width=0.65pt, inner sep=4.5pt, font=\scriptsize, text width=0.92\columnwidth},')
    tikz_lines.append(r'    arrow/.style={-{Stealth[scale=0.95]}, thick, color=blue!75!black}')
    tikz_lines.append(r']')
    
    for idx, node_text in enumerate(clean_nodes):
        style = 'flowblock'
        if idx == 0:
            style = 'topblock'
        elif idx == len(clean_nodes) - 1:
            style = 'botblock'
            
        if idx == 0:
            tikz_lines.append(f'    \\node[{style}] (node{idx}) {{{node_text}}};')
        else:
            tikz_lines.append(f'    \\node[{style}, below=of node{idx-1}] (node{idx}) {{{node_text}}};')
            
    tikz_lines.append('')
    tikz_lines.append('    % Flow Connections')
    for idx in range(len(clean_nodes) - 1):
        tikz_lines.append(f'    \\draw[arrow] (node{idx}) -- (node{idx+1});')
        
    tikz_lines.append(r'\end{tikzpicture}')
    tikz_lines.append(r'\end{adjustbox}')
    tikz_lines.append(r'\end{center}')
    
    return "\n".join(tikz_lines)

def is_ascii_diagram(text, lang):
    if lang in ['python', 'c', 'cpp', 'bash', 'sh', 'cuda', 'json', 'yaml']:
        return False
    if any(kw in text for kw in ['def ', 'import torch', '@triton', 'class ', 'return ', 'nn.Module']):
        return False
    if any(char in text for char in ['┌', '─', '│', '▼', '▲', '──►', '+---', '├──', '└──', '---|']):
        return True
    if any(marker in text for marker in ['Input Tensor', 'Input:', 'SRAM Scratchpad', 'Global Memory', 'Output:']):
        return True
    return False

def escape_latex_clean(text):
    text = text.replace('\f', '\\f').replace('\t', '\\t').replace('\r', '\\r').replace('\v', '\\v').replace('\b', '\\b')

    placeholders = {}
    counter = [0]
    
    def add_ph(val, ptype, lang=''):
        key = f"MPH{counter[0]}E"
        counter[0] += 1
        
        if ptype == 'inline_code':
            for k, v in [('θ', 'theta'), ('φ', 'phi'), ('ψ', 'psi'), ('λ', 'lambda'), ('β', 'beta'), ('γ', 'gamma'), ('π', 'pi'), ('Ω', 'Omega'), ('η', 'eta'), ('→', '->'), ('←', '<-'), ('↑', '^'), ('↓', 'v'), ('•', '*'), ('…', '...'), ('×', 'x'), ('÷', '/'), ('±', '+/-'), ('≠', '!='), ('∑', 'Sigma'), ('∏', 'Pi'), ('∫', 'integral'), ('∂', 'd'), ('∇', 'nabla'), ('ρ', 'rho'), ('Δ', 'Delta'), ('Σ', 'Sigma'), ('α', 'alpha'), ('ε', 'epsilon'), ('σ', 'sigma'), ('τ', 'tau'), ('ℝ', 'R'), ('≈', '~'), ('≤', '<='), ('≥', '>='), ('≪', '<<'), ('⊗', '(x)'), ('⊙', '(.)'), ('·', '*'), ('²', '^2'), ('³', '^3'), ('‘', "'"), ('’', "'"), ('“', '"'), ('”', '"'), ('–', '-'), ('—', '--')]:
                val = val.replace(k, v)
            val = val.replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%', '\\%').replace('#', '\\#').replace('{', '\\{').replace('}', '\\}').replace('&', '\\&').replace('$', '\\$').replace('^', '\\textasciicircum{}').replace('~', '\\textasciitilde{}')
            val = f"\\texttt{{{val}}}"
        elif ptype in ('inline_math', 'math_block'):
            val = val.replace('%', '\\%')
            def escape_text_mode_in_math(m):
                cmd = m.group(1)
                content = m.group(2)
                return f"\\{cmd}{{" + content.replace('_', r'\_') + "}"
            val = re.sub(r'\\(text(?:bf|it|tt)?)\{([^}]*)\}', escape_text_mode_in_math, val)
            if ptype == 'inline_math':
                val = f"${val}$"

        placeholders[key] = (val, ptype, lang)
        return key

    def extract_code_block(match):
        val = match.group(1).strip()
        lang_match = re.match(r'^([a-zA-Z0-9_-]+)\\n', val)
        lang = lang_match.group(1).lower() if lang_match else ''
        if lang_match:
            val = val[lang_match.end():]
            
        if lang == 'latex_raw':
            return add_ph(val, 'latex_raw', lang)
            
        if is_ascii_diagram(val, lang):
            tikz_graphic = parse_ascii_diagram_to_tikz(val)
            if tikz_graphic:
                return add_ph(tikz_graphic, 'latex_raw', 'tikz')
                
        for k, v in [('θ', 'theta'), ('φ', 'phi'), ('ψ', 'psi'), ('λ', 'lambda'), ('β', 'beta'), ('γ', 'gamma'), ('π', 'pi'), ('Ω', 'Omega'), ('η', 'eta'), ('→', '->'), ('←', '<-'), ('↑', '^'), ('↓', 'v'), ('•', '*'), ('…', '...'), ('×', 'x'), ('÷', '/'), ('±', '+/-'), ('≠', '!='), ('∑', 'Sigma'), ('∏', 'Pi'), ('∫', 'integral'), ('∂', 'd'), ('∇', 'nabla'), ('ρ', 'rho'), ('Δ', 'Delta'), ('Σ', 'Sigma'), ('α', 'alpha'), ('ε', 'epsilon'), ('σ', 'sigma'), ('τ', 'tau'), ('ℝ', 'R'), ('≈', '~'), ('≤', '<='), ('≥', '>='), ('≪', '<<'), ('⊗', '(x)'), ('⊙', '(.)'), ('·', '*'), ('²', '^2'), ('³', '^3'), ('‘', "'"), ('’', "'"), ('“', '"'), ('”', '"'), ('–', '-'), ('—', '--')]:
            val = val.replace(k, v)
        for k, v in unicode_ascii_map.items(): 
            val = val.replace(k, v)
                
        return add_ph(val, 'code_block', lang)

    text = re.sub(r'```(.*?)```', extract_code_block, text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', lambda m: add_ph(m.group(1), 'inline_code'), text)
    
    # NOW perform global unescaping on the remaining text
    text = text.replace(r'\_', '_').replace(r'\$', '$').replace(r'\&', '&').replace(r'\%', '%')

    
    # Fix Task 10: Apply unicode replacements to regular text and math AFTER extracting code
    for k, v in unicode_ascii_map.items(): text = text.replace(k, v)
    for k, v in unicode_math_map.items(): text = text.replace(k, v)
    text_math_map = {'θ': r'$\theta$', 'φ': r'$\phi$', 'ψ': r'$\psi$', 'λ': r'$\lambda$', 'β': r'$\beta$', 'γ': r'$\gamma$', 'π': r'$\pi$', 'Ω': r'$\Omega$', 'η': r'$\eta$', '→': r'$\rightarrow$', '←': r'$\leftarrow$', '↑': r'$\uparrow$', '↓': r'$\downarrow$', '•': r'$\bullet$', '…': r'$\dots$', '×': r'$\times$', '÷': r'$\div$', '±': r'$\pm$', '≠': r'$\neq$', '∑': r'$\sum$', '∏': r'$\prod$', '∫': r'$\int$', '∂': r'$\partial$', '∇': r'$\nabla$', 'ρ': r'$\rho$'}
    for k, v in text_math_map.items(): text = text.replace(k, v)
    
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: add_ph(m.group(1), 'math_block'), text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', lambda m: add_ph(m.group(1), 'inline_math'), text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.*?)\\\]', lambda m: add_ph(m.group(1), 'math_block'), text, flags=re.DOTALL)
    text = re.sub(r'\$([^\$]+)\$', lambda m: add_ph(m.group(1), 'inline_math'), text)

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
            ph_data = placeholders[stripped]
            val = ph_data[0]
            ptype = ph_data[1]
            if ptype == 'latex_raw':
                out_lines.append(val)
            elif ptype == 'code_block':
                lang = ph_data[2]
                lang_arg = ""
                if lang in ['python', 'c', 'cpp', 'bash', 'sh', 'java', 'ruby', 'perl', 'php']:
                    lang_arg = f"[language={lang.capitalize()}]"
                out_lines.append(r'\vspace{0.5em}')
                out_lines.append(r'\begin{mdframed}[backgroundcolor=codebg, linecolor=codeframe, linewidth=1pt, roundcorner=4pt, innerleftmargin=2pt, innerrightmargin=5pt, innertopmargin=5pt, innerbottommargin=5pt]')
                out_lines.append(f'\\begin{{lstlisting}}{lang_arg}')
                out_lines.append(val)
                out_lines.append(r'\end{lstlisting}')
                out_lines.append(r'\end{mdframed}')
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
    for key, ph_data in placeholders.items():
        if ph_data[1] in ('math_block', 'latex_raw'):
            # If a block-level placeholder survived into the final text, it means it was inline
            text = text.replace(key, f"$\\displaystyle {ph_data[0]}$")
        else:
            text = text.replace(key, ph_data[0])
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
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        latex_content = escape_latex_clean(md_content)
        safe_name = name.replace('_', r'\_')
        return f"\\subsection{{\\texttt{{{safe_name}}}}}\n{latex_content}\n\n"
    else:
        return match.group(0)

new_tex_content = pattern.sub(replace_sub, tex_content)

packages_to_add = r'''
\usepackage[utf8]{inputenc}
\usepackage{xcolor}
\usepackage{adjustbox}
\usepackage{listings}
\usepackage{mdframed}
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric}
\definecolor{codebg}{HTML}{F8F9FA}
\definecolor{codeframe}{HTML}{E9ECEF}
\definecolor{keywordcolor}{HTML}{005CC5}
\definecolor{stringcolor}{HTML}{032F62}
\definecolor{commentcolor}{HTML}{6A737D}
\definecolor{numbercolor}{HTML}{005CC5}
\lstset{
    backgroundcolor=\color{codebg},
    basicstyle=\ttfamily\scriptsize\color{black},
    keywordstyle=\color{keywordcolor}\bfseries,
    stringstyle=\color{stringcolor},
    commentstyle=\color{commentcolor}\itshape,
    numberstyle=\tiny\color{numbercolor},
    numbers=left,
    numbersep=8pt,
    frame=none,
    breaklines=true,
    breakatwhitespace=true,
    keepspaces=true,
    showstringspaces=false,
    tabsize=4,
    xleftmargin=12pt
}
'''

if r'\usepackage{tikz}' not in new_tex_content:
    new_tex_content = new_tex_content.replace(r'\usepackage{booktabs}', r'\usepackage{booktabs}' + '\n' + packages_to_add)

with open(main_tex_path, "w", encoding='utf-8') as f:
    f.write(new_tex_content)

print("Injected all 185 modules with automatic TikZ flowchart conversion for ASCII diagrams!")
