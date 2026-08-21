import glob
import re

files = glob.glob("Codes/framework/attentions/*.py")
count = 0
for f in files:
    with open(f, 'r') as file:
        content = file.read()
    
    # Replace tiny epsilons with 1e-5
    # e.g., 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12
    # but be careful not to replace something inside a string if it doesn't matter (actually strings are fine)
    new_content = re.sub(r'1e-(7|8|9|10|11|12)', '1e-5', content)
    
    if new_content != content:
        with open(f, 'w') as file:
            file.write(new_content)
        count += 1

print(f"Patched epsilons in {count} files.")
