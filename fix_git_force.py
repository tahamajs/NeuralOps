import re

path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\run_suite.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace(
    'subprocess.run(["git", "add", str(RESULTS_DIR.parent)], cwd=str(PROJECT_DIR), check=True)',
    'subprocess.run(["git", "add", "-f", str(RESULTS_DIR)], cwd=str(PROJECT_DIR), check=True)'
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
