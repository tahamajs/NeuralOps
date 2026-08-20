import re
path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    "const res = await fetch('http://192.168.1.100:8080/api/stats');",
    "const res = await fetch('/api/stats');"
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
