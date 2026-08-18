import re

with open('clean_latex_injection.py', 'r') as f:
    text = f.read()

# Add extraction for $$ ... $$ and \( ... \)
replacement = """    text = re.sub(r'\\$\\$(.*?)\\$\\$', lambda m: add_ph(m.group(1), 'math_block'), text, flags=re.DOTALL)
    text = re.sub(r'\\\\\((.*?)\\\\\)', lambda m: add_ph(m.group(1), 'inline_math'), text, flags=re.DOTALL)
    text = re.sub(r'\\\\\\[(.*?)\\\\\\]', lambda m: add_ph(m.group(1), 'math_block'), text, flags=re.DOTALL)
    text = re.sub(r'\\$([^\\$]+)\\$', lambda m: add_ph(m.group(1), 'inline_math'), text)"""

text = re.sub(r"    text = re\.sub\(r'\\\\\\\[\(\.\*\?\)\\\\\\]', lambda m: add_ph\(m\.group\(1\), 'math_block'\), text, flags=re\.DOTALL\)\n    text = re\.sub\(r'\\\$\[\^\\\$\]\+\\\$', lambda m: add_ph\(m\.group\(1\), 'inline_math'\), text\)", replacement, text)

with open('clean_latex_injection.py', 'w') as f:
    f.write(text)
