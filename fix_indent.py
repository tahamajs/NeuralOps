import glob
import os

def fix_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    changed = False
    for i in range(len(lines) - 2):
        if "except Exception as e:" in lines[i]:
            # find the indentation of 'except'
            indent = len(lines[i]) - len(lines[i].lstrip())
            
            # The next lines might be the print and return that are wrongly indented
            # Usually they are indented exactly the same as 'except' because of bad find-and-replace
            # or maybe less. Let's just check if lines[i+1] starts with 'print(f"copy_weights' and lacks proper indent
            
            # Count indent of next line
            next_indent = len(lines[i+1]) - len(lines[i+1].lstrip())
            if next_indent <= indent and "print(" in lines[i+1]:
                # Fix it by indenting it 4 spaces more than 'except'
                lines[i+1] = (" " * (indent + 4)) + lines[i+1].lstrip()
                changed = True
                
            if "return None" in lines[i+2]:
                next_next_indent = len(lines[i+2]) - len(lines[i+2].lstrip())
                if next_next_indent <= indent:
                    lines[i+2] = (" " * (indent + 4)) + lines[i+2].lstrip()
                    changed = True

    if changed:
        with open(filepath, 'w') as f:
            f.writelines(lines)
            
for f in glob.glob("Codes/framework/attentions/*.py") + glob.glob("Codes/framework/compression_engine/*.py"):
    fix_file(f)

