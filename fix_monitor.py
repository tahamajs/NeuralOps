import re

path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace('WORKDIR / "results"', 'WORKDIR / "experiments"')

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
