import re
path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\run_suite.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix batch size logic for better utilization
content = re.sub(
    r'cfg\.batch_size = 16 if attn in \("full", "flash3"\) else 32',
    'if attn in ("full", "flash3", "griffin_ssd", "aegis_v3", "spectra_v2", "samba_m3"):\n                cfg.batch_size = 16\n            elif attn in ("svd_linear", "performer", "linformer", "nystrom"):\n                cfg.batch_size = 64\n            else:\n                cfg.batch_size = 32',
    content
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
