import re
path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\run_suite.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Update TASKS
content = re.sub(
    r'TASKS = \["sst2", "mrpc", "qnli"\]',
    'TASKS = ["sst2", "mrpc", "qnli", "cola", "qqp", "rte", "stsb", "wnli"]',
    content
)

# Update OOM logic
content = content.replace(
    'if attn in ["full", "flash3"]:',
    'if attn in ["full", "flash3", "griffin_ssd", "aegis_v3", "spectra_v2", "samba_m3"]:'
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
