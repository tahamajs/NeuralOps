import re

path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\run_suite.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace: cfg.batch_size = 32
# With: cfg.batch_size = 16 if attn in ("full", "flash3") else 32

content = re.sub(
    r"cfg\.batch_size\s*=\s*32",
    "cfg.batch_size = 16 if attn in (\"full\", \"flash3\") else 32",
    content
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
