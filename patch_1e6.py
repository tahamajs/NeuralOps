import glob
import re

files = glob.glob("Codes/framework/attentions/*.py")
count = 0
for f in files:
    with open(f, 'r') as file:
        content = file.read()
    
    # Replace + 1e-6 or min=1e-6 or > 1e-6 with 1e-5
    new_content = re.sub(r'(\+|min=|>\s*|max=|eps=)\s*1e-6', r'\1 1e-5', content)
    
    if new_content != content:
        with open(f, 'w') as file:
            file.write(new_content)
        count += 1

print(f"Patched 1e-6 in {count} files.")
