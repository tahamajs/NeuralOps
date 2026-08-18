import re

with open('main.tex', 'r') as f:
    text = f.read()

def escape_text_mode(m):
    cmd = m.group(1)
    content = m.group(2)
    # unescape first to avoid double escape
    content = content.replace(r'\_', '_')
    return '\\\\' + cmd + '{' + content.replace('_', r'\_') + '}'

text = re.sub(r'\\(text(?:bf|it|tt)?)\{([^}]*)\}', escape_text_mode, text)

with open('main.tex', 'w') as f:
    f.write(text)
