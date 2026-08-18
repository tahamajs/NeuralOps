import re

with open('clean_latex_injection.py', 'r') as f:
    text = f.read()

replacement = """    placeholders = {}
    counter = [0]
    
    def add_ph(val, ptype, lang=''):
        key = f"MPH{counter[0]}E"
        counter[0] += 1
        
        if ptype == 'inline_code':
            val = val.replace('\\\\', '\\\\textbackslash{}').replace('_', '\\\\_').replace('%', '\\\\%').replace('#', '\\\\#').replace('{', '\\\\{').replace('}', '\\\\}').replace('&', '\\\\&').replace('$', '\\\\$').replace('^', '\\\\textasciicircum{}').replace('~', '\\\\textasciitilde{}')
            val = f"\\\\texttt{{{val}}}"
        elif ptype in ('inline_math', 'math_block'):
            val = val.replace('%', '\\\\%')
            def escape_text_mode_in_math(m):
                cmd = m.group(1)
                content = m.group(2)
                return f"\\\\{cmd}{{" + content.replace('_', r'\\_') + "}"
            val = re.sub(r'\\\\(text(?:bf|it|tt)?)\\{([^}]*)\\}', escape_text_mode_in_math, val)
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
                
        for k, v in [('θ', 'theta'), ('φ', 'phi'), ('ψ', 'psi'), ('λ', 'lambda'), ('β', 'beta'), ('γ', 'gamma'), ('π', 'pi'), ('Ω', 'Omega'), ('η', 'eta'), ('→', '->'), ('←', '<-'), ('↑', '^'), ('↓', 'v'), ('•', '*'), ('…', '...'), ('×', 'x'), ('÷', '/'), ('±', '+/-'), ('≠', '!='), ('∑', 'Sigma'), ('∏', 'Pi'), ('∫', 'integral'), ('∂', 'd'), ('∇', 'nabla'), ('ρ', 'rho'), ('Δ', 'Delta'), ('Σ', 'Sigma'), ('α', 'alpha'), ('ε', 'epsilon'), ('σ', 'sigma'), ('τ', 'tau'), ('ℝ', 'R'), ('≈', '~'), ('≤', '<='), ('≥', '>='), ('≪', '<<'), ('⊗', '(x)'), ('⊙', '(.)')]:
            val = val.replace(k, v)
        for k, v in unicode_ascii_map.items(): 
            val = val.replace(k, v)
                
        return add_ph(val, 'code_block', lang)

    text = re.sub(r'```(.*?)```', extract_code_block, text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', lambda m: add_ph(m.group(1), 'inline_code'), text)
    
    # NOW perform global unescaping on the remaining text
    text = text.replace(r'\\_', '_').replace(r'\\$', '$').replace(r'\\&', '&').replace(r'\\%', '%')
"""

target = re.compile(r"    # Fix Task 2: Remove manual LLM escapes to prevent double escaping artifacts\n    text = text\.replace\(r'\\_', '_'\)\.replace\(r'\\\$', '\$'\)\.replace\(r'\\&', '&'\)\.replace\(r'\\%', '%'\)\n        \n    placeholders = \{\}\n    counter = \[0\].*?    text = re\.sub\(r'`\(\.\*\?\)`', lambda m: add_ph\(m\.group\(1\), 'inline_code'\), text\)", re.DOTALL)
text = target.sub(replacement, text)

with open('clean_latex_injection.py', 'w') as f:
    f.write(text)
