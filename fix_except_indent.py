import os
import glob
files = glob.glob("Codes/framework/attentions/*.py") + glob.glob("Codes/framework/compression_engine/*.py")
for file in files:
    with open(file, "r") as f:
        lines = f.readlines()
    changed = False
    for i in range(len(lines) - 2):
        if "except Exception as e:" in lines[i]:
            indent = len(lines[i]) - len(lines[i].lstrip())
            next_line = lines[i+1]
            if "print(f'copy_weights failed: {e}')" in next_line:
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= indent:
                    lines[i+1] = (" " * (indent + 4)) + next_line.lstrip()
                    changed = True
            next_next_line = lines[i+2]
            if "return None" in next_next_line:
                next_next_indent = len(next_next_line) - len(next_next_line.lstrip())
                if next_next_indent <= indent:
                    lines[i+2] = (" " * (indent + 4)) + next_next_line.lstrip()
                    changed = True
    if changed:
        with open(file, "w") as f:
            f.writelines(lines)
        print(f"Fixed {file}")
