import os
import glob
import re

files = glob.glob("Codes/framework/attentions/*.py")
count = 0
for f in files:
    with open(f, 'r') as file:
        content = file.read()
    
    new_content = re.sub(r'-(1e9|1e4)', '-10000.0', content)
    new_content = re.sub(r'float\("-1e9"\)', '-10000.0', new_content)
    
    if new_content != content:
        with open(f, 'w') as file:
            file.write(new_content)
        count += 1

print(f"Patched {count} files.")
