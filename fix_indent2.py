path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "self.end_headers()" in line:
        # get indent of previous line
        prev = lines[i-1]
        indent = len(prev) - len(prev.lstrip(" \t"))
        lines[i] = " " * indent + line.lstrip(" \t")

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
