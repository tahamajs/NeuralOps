import re
import glob
import os

def clean_label_for_latex(text):
    # Escape LaTeX special chars except already formatted math
    text = text.replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
    # Clean up box characters
    for ch in ['┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '─', '│', '►', '▼', '◄', '▲', '+---', '|---', '+']:
        text = text.replace(ch, '')
    text = text.strip()
    # Format shapes [B, N, D] into math
    text = re.sub(r'\[([BNDHLdkvr\d\s,\*\+_]+)\]', r'$[\1]$', text)
    # Fix underscores outside math
    parts = text.split('$')
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace('_', r'\_')
    text = '$'.join(parts)
    # Basic math mappings
    text = text.replace('->', r'$\rightarrow$').replace('-->', r'$\longrightarrow$')
    text = text.replace('>=', r'$\ge$').replace('<=', r'$\le$')
    return text

def parse_ascii_diagram_to_tikz(ascii_text):
    lines = [l.rstrip() for l in ascii_text.strip().split('\n') if l.strip()]
    if not lines:
        return None
    
    # Extract distinct blocks / steps from the ASCII diagram
    nodes = []
    current_block_lines = []
    
    for line in lines:
        stripped = line.strip()
        # If line is mostly horizontal border or arrows
        if re.match(r'^[┌└├─\+\=\-\s]+$', stripped) or re.match(r'^[│\|\s▼▲►◄vV\^\|]+$', stripped) and not any(c.isalnum() for c in stripped):
            if current_block_lines:
                raw_text = " ".join([l.strip('│| ') for l in current_block_lines if l.strip('│| ')])
                if raw_text:
                    nodes.append(raw_text)
                current_block_lines = []
            continue
        
        # Check if line contains actual text
        content = re.sub(r'^[┌└├│\|\+\-\s]+', '', line)
        content = re.sub(r'[┐┘┤│\|\+\-\s]+$', '', content).strip()
        if content:
            current_block_lines.append(content)
            
    if current_block_lines:
        raw_text = " ".join([l.strip('│| ') for l in current_block_lines if l.strip('│| ')])
        if raw_text:
            nodes.append(raw_text)
            
    # Filter and clean nodes
    clean_nodes = []
    for n in nodes:
        cl = clean_label_for_latex(n)
        if cl and len(cl) > 1 and not re.match(r'^[vV\^\|\-\+\>\<\s]+$', cl):
            # break long text into formatted lines with \\
            if len(cl) > 50 and '\\\\' not in cl:
                # split at logical boundaries
                cl = re.sub(r'(\s*:\s*)', r':\\\\ ', cl, count=1)
            clean_nodes.append(cl)
            
    if len(clean_nodes) < 2:
        return None
        
    # Generate TikZ code
    tikz_lines = []
    tikz_lines.append(r'\begin{center}')
    tikz_lines.append(r'\begin{adjustbox}{max width=0.98\columnwidth}')
    tikz_lines.append(r'\begin{tikzpicture}[')
    tikz_lines.append(r'    node distance=0.85cm,')
    tikz_lines.append(r'    every node/.style={align=center},')
    tikz_lines.append(r'    flowblock/.style={rectangle, draw=blue!70!black, fill=blue!4, rounded corners=3pt, line width=0.65pt, inner sep=4.5pt, font=\scriptsize, text width=0.92\columnwidth},')
    tikz_lines.append(r'    topblock/.style={rectangle, draw=teal!70!black, fill=teal!4, rounded corners=3pt, line width=0.65pt, inner sep=4.5pt, font=\scriptsize, text width=0.92\columnwidth},')
    tikz_lines.append(r'    botblock/.style={rectangle, draw=indigo!70!black, fill=indigo!4, rounded corners=3pt, line width=0.65pt, inner sep=4.5pt, font=\scriptsize, text width=0.92\columnwidth},')
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

# Test on some sample files
sample_files = glob.glob('/Users/tahamajs/Documents/uni/Project/Docs/content/attentions/*.md')[:5]
for f in sample_files:
    with open(f, 'r', encoding='utf-8') as fp:
        c = fp.read()
    blocks = re.findall(r'```(.*?)```', c, re.DOTALL)
    for b in blocks:
        if any(char in b for char in ['┌', '─', '│', '▼', '──►', '+---', '├──']):
            if not any(kw in b for kw in ['def ', 'import torch', '@triton', 'class ']):
                print(f"=== Testing File: {os.path.basename(f)} ===")
                res = parse_ascii_diagram_to_tikz(b)
                if res:
                    print(res[:350] + "\n...")
                    print("SUCCESS\n")
