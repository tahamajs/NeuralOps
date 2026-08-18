import re

with open('main.tex', 'r') as f:
    text = f.read()

def escape_text_mode(m):
    cmd = m.group(1)
    content = m.group(2)
    # Ensure any _ inside \text{...} is properly escaped.
    # First unescape to avoid double escaping, then escape.
    content = content.replace(r'\_', '_')
    content = content.replace('_', r'\_')
    return f"\\{cmd}{{{content}}}"

# We only want to match \text, \textbf, \textit, \texttt
text = re.sub(r'\\(text(?:bf|it|tt)?)\{([^}]*)\}', escape_text_mode, text)

with open('main.tex', 'w') as f:
    f.write(text)
