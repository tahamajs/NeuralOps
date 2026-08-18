import re

text = "**O(N²·D)**"
unicode_math_map = {'²': r'$^2$', '·': r'$\cdot$ '}
for k, v in unicode_math_map.items(): text = text.replace(k, v)

print(text)

counter = [0]
placeholders = {}
def add_ph(val, ptype, lang=''):
    key = f"MPH{counter[0]}E"
    counter[0] += 1
    if ptype == 'inline_math':
        val = f"${val}$"
    placeholders[key] = (val, ptype, lang)
    return key

text = re.sub(r'\$([^\$]+)\$', lambda m: add_ph(m.group(1), 'inline_math'), text)

print(text)

for key, ph_data in placeholders.items():
    text = text.replace(key, ph_data[0])
    
print(text)
