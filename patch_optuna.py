import re

path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\hyperopt\backends\optuna_optimizer.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_import = "import optuna"
new_import = """try:
    import optuna
except ImportError:
    optuna = None"""

content = content.replace(old_import, new_import)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patched optuna import fallback!")
