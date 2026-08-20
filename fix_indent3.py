path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "self.send_header(\"Access-Control-Allow-Methods\", \"GET, OPTIONS\")" in line or \
       "self.send_header(\"Access-Control-Allow-Headers\", \"*\")" in line:
        prev = lines[i-1]
        indent = len(prev) - len(prev.lstrip(" \t"))
        if indent > 0:
            lines[i] = " " * indent + line.lstrip(" \t")

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
