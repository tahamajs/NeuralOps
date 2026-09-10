import re

filepath = r"D:\SVD_Project\Codes\framework\run_quality_sweep.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Add torch.cuda.set_per_process_memory_fraction(0.9) if not present
if "set_per_process_memory_fraction" not in content:
    content = re.sub(
        r"import torch\n",
        r"import torch\nif torch.cuda.is_available():\n    torch.cuda.set_per_process_memory_fraction(0.9, 0)\n",
        content
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("Patched run_quality_sweep.py memory limit.")
else:
    print("Memory limit already present.")
