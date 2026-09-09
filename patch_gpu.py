import os, glob, re

target_dirs = [
    r"D:\SVD_Project\Codes\framework\attentions",
    r"D:\SVD_Project\Codes\framework\compression_engine"
]

patched_count = 0

for d in target_dirs:
    for filepath in glob.glob(os.path.join(d, "*.py")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        
        # 1. Fix except Exception: pass in inject functions to return None
        # This prevents silently returning randomly initialized modules.
        content = re.sub(r"except Exception:\s+pass", r"except Exception as e:\n                print(f'copy_weights failed: {e}')\n                return None", content)

        # 2. Add output_scale to __init__ if missing
        # We look for super().__init__(...)
        if "self.output_scale" not in content and "super().__init__" in content:
            content = re.sub(
                r"(super\(\)\.__init__\([^)]*\)\s*\n(\s+))", 
                r"\1if not hasattr(self, 'output_scale'):\n\2    import torch, torch.nn as nn\n\2    self.output_scale = nn.Parameter(torch.ones(1))\n\2", 
                content
            )

        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            patched_count += 1
            print(f"Patched: {os.path.basename(filepath)}")

print(f"Total files patched: {patched_count}")
