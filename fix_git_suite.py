import re
path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\run_suite.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace('RESULTS_DIR = BASE_DIR / "results"', 'RESULTS_DIR = BASE_DIR / "experiments"')
content = content.replace('RESULTS_DIR.mkdir(parents=True, exist_ok=True)', 'RESULTS_DIR.mkdir(parents=True, exist_ok=True)')

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
