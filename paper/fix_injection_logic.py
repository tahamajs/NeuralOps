import re

with open('clean_latex_injection.py', 'r') as f:
    text = f.read()

# Remove the late processing loop
late_processing_regex = r"    for key, ph_data in placeholders\.items\(\):\n.*?(?=\n    return text\n)"
text = re.sub(late_processing_regex, "", text, flags=re.DOTALL)

# Add early processing inside add_ph
replacement = """def add_ph(val, ptype):
    global ph_counter
    key = f"MPH{ph_counter}E"
    
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

    placeholders[key] = (val, ptype)
    ph_counter += 1
    return key"""

text = re.sub(r"def add_ph\(val, ptype\):\n    global ph_counter\n    key = f\"MPH\{ph_counter\}E\"\n    placeholders\[key\] = \(val, ptype\)\n    ph_counter \+= 1\n    return key", replacement, text)

with open('clean_latex_injection.py', 'w') as f:
    f.write(text)
