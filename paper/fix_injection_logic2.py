import re

with open('clean_latex_injection.py', 'r') as f:
    text = f.read()

replacement = """
    text = '\\n'.join(out_lines)
    for key, ph_data in placeholders.items():
        text = text.replace(key, ph_data[0])
    return text
"""

text = re.sub(r"    text = '\\\\n'\.join\(out_lines\)\n    return text", replacement, text)

with open('clean_latex_injection.py', 'w') as f:
    f.write(text)
