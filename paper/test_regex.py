import re
val = r'\text{if } \text{use_full_attention} = \text{False}'
def escape_text_mode_in_math(m):
    cmd = m.group(1)
    content = m.group(2)
    return f"\\{cmd}{{" + content.replace('_', r'\_') + "}"
val = re.sub(r'\\(text(?:bf|it|tt)?)\{([^}]*)\}', escape_text_mode_in_math, val)
print(repr(val))
