path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    stripped = line.lstrip(" \t")
    if stripped.startswith("self.send_header(") or stripped.startswith("self.end_headers()") or stripped.startswith("self.wfile."):
        # Check surrounding context for proper indentation
        prev_idx = i - 1
        while prev_idx >= 0 and lines[prev_idx].strip() == "":
            prev_idx -= 1
        
        if prev_idx >= 0:
            prev_line = lines[prev_idx]
            indent = len(prev_line) - len(prev_line.lstrip(" \t"))
            
            # If the previous line was if/for/elif/else/def, the indent should be +4
            if prev_line.lstrip(" \t").startswith(("if ", "elif ", "else:", "def ", "for ")):
                indent += 4
                
            lines[i] = " " * indent + stripped

with open(path, "w", encoding="utf-8") as f:
    f.writelines(lines)
