path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "self.send_header(\"Access-Control-Allow-Methods\", \"GET, OPTIONS\")" in line:
        lines[i] = line.lstrip(" \t")
        # find indent of previous line
        prev_indent = len(lines[i-1]) - len(lines[i-1].lstrip(" \t"))
        if lines[i-1].strip() == "":
            prev_indent = len(lines[i-2]) - len(lines[i-2].lstrip(" \t"))
        lines[i] = " " * prev_indent + lines[i]

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
