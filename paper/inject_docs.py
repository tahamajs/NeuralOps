import json
import re
import os

docs_dir = "/Users/tahamajs/Documents/uni/Project/Docs/"
main_tex_path = "/Users/tahamajs/Documents/uni/Project/paper/main.tex"

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

def escape_latex(text):
    placeholders = {}
    counter = [0]
    
    def replacer(match):
        idx = counter[0]
        counter[0] += 1
        key = f"MATHPLACEHOLDER{idx}END"
        placeholders[key] = match.group(0)
        return key

    # Replace unicodes
    for k, v in unicode_ascii_map.items():
        text = text.replace(k, v)
        
    # Math unicodes should be replaced inside math blocks mainly, but we can just replace them everywhere
    # because they are mostly used in math context.
    for k, v in unicode_math_map.items():
        text = text.replace(k, f"${v}$")

    # Protect code blocks
    text = re.sub(r'```(.*?)```', replacer, text, flags=re.DOTALL)
    # Protect inline code
    text = re.sub(r'`(.*?)`', replacer, text)
    # Protect display math
    text = re.sub(r'\$\$(.*?)\$\$', replacer, text, flags=re.DOTALL)
    text = re.sub(r'\\\[(.*?)\\\]', replacer, text, flags=re.DOTALL)
    # Protect begin/end environments if any
    text = re.sub(r'\\begin\{(.*?)\}(.*?)\\end\{\1\}', replacer, text, flags=re.DOTALL)
    # Protect inline math
    text = re.sub(r'\$(.*?)\$', replacer, text)
    
    # Protect markdown tables (header |...| \n |---| )
    def table_replacer(match):
        idx = counter[0]
        counter[0] += 1
        key = f"MATHPLACEHOLDER{idx}END"
        placeholders[key] = "\\begin{verbatim}\n" + match.group(0) + "\n\\end{verbatim}"
        return key
    text = re.sub(r'(^\|.*\|$\n^\|[-: \|]+\|$\n(?:^\|.*\|$\n?)*)', table_replacer, text, flags=re.MULTILINE)

    # Escape latex specials
    text = text.replace('\\', '\\textbackslash{}')
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    text = text.replace('#', '\\#')
    text = text.replace('_', '\\_')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    text = text.replace('~', '\\textasciitilde{}')
    text = text.replace('^', '\\textasciicircum{}')
    text = text.replace('$', '\\$') # Escape unmatched $

    # Formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'\*(.*?)\*', r'\\textit{\1}', text)
    
    text = re.sub(r'^### (.*)', r'\\subsubsection*{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*)', r'\\subsection*{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.*)', r'\\subsection*{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*]\s+(.*)', r'\\textbullet{} \1\n', text, flags=re.MULTILINE)

    # Restore placeholders
    for key, p in placeholders.items():
        if p.startswith('```'):
            code = p[3:-3].strip()
            code = re.sub(r'^[a-zA-Z0-9]+\n', '', code)
            p = f"\\begin{{verbatim}}\n{code}\n\\end{{verbatim}}"
        elif p.startswith('`') and not p.startswith('```'):
            inner = p[1:-1]
            inner = inner.replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%', '\\%').replace('#', '\\#').replace('{', '\\{').replace('}', '\\}').replace('&', '\\&').replace('$', '\\$')
            p = f"\\texttt{{{inner}}}"
        elif p.startswith('$$'):
            inner = p[2:-2].strip()
            inner = re.sub(r'\n\s*\n', '\n', inner)
            p = f"\\begin{{equation*}}\n{inner}\n\\end{{equation*}}"
        elif p.startswith('\\['):
            inner = p[2:-2].strip()
            inner = re.sub(r'\n\s*\n', '\n', inner)
            p = f"\\begin{{equation*}}\n{inner}\n\\end{{equation*}}"
        elif p.startswith('\\begin{verbatim}'):
            pass # Keep as is for tables
            
        text = text.replace(key, p)
        
    return text

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
        latex_content = escape_latex(md_content)
        return f"\\subsection{{\\texttt{{{name}}}}}\n{latex_content}\n\n"
    else:
        return match.group(0)

# We must start from the original main.tex to avoid double-escaping if we run this script again.
# The original main.tex had \subsection{\texttt{name}} followed by simple text.
# Wait, since we already replaced the content in main.tex in the previous run, running the regex again will match the CURRENT content and replace it!
# Because the regex matches \subsection{\texttt{...}} up to the next subsection, it will cleanly overwrite the old injected content with the new injected content.
new_tex_content = pattern.sub(replace_sub, tex_content)

new_tex_content = new_tex_content.replace(r'\usepackage{microtype}', r'%\usepackage{microtype}')

with open(main_tex_path, "w") as f:
    f.write(new_tex_content)
print("Conversion complete.")
