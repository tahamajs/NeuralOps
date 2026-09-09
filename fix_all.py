import os, glob, re

target_dirs = [
    r"./Codes/framework/attentions",
    r"./Codes/framework/compression_engine"
]

patched_count = 0

for d in target_dirs:
    for filepath in glob.glob(os.path.join(d, "*.py")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content
        
        # 1. Fix exception handling
        content = re.sub(r"except Exception:\s+pass", r"except Exception as e:\n                print(f'copy_weights failed: {e}')\n                return None", content)
        
        # 2. Add output_scale to __init__
        if "self.output_scale" not in content and "super().__init__" in content:
            content = re.sub(
                r"(super\(\)\.__init__\([^)]*\)\s*\n(\s+))", 
                r"\1if not hasattr(self, 'output_scale'):\n\2    import torch, torch.nn as nn\n\2    self.output_scale = nn.Parameter(torch.ones(1))\n\2", 
                content
            )
            
            # 3. Add output_scale multiplication inside forward
            # Safely using regex on the entire file content
            # To be safe, we only replace `return ` if it's returning output or out or context
            # We must be careful about indentation and other returns
            replacements = [
                (r"(\n\s*)return output, None", r"\1return output * self.output_scale, None"),
                (r"(\n\s*)return output(\n)", r"\1return output * self.output_scale\2"),
                (r"(\n\s*)return out, None", r"\1return out * self.output_scale, None"),
                (r"(\n\s*)return out(\n)", r"\1return out * self.output_scale\2"),
                (r"(\n\s*)return context, None", r"\1return context * self.output_scale, None"),
                (r"(\n\s*)return context(\n)", r"\1return context * self.output_scale\2"),
                (r"(\n\s*)return ctx, None", r"\1return ctx * self.output_scale, None"),
                (r"(\n\s*)return ctx(\n)", r"\1return ctx * self.output_scale\2"),
            ]
            for old, new in replacements:
                content = re.sub(old, new, content)

        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            patched_count += 1
            print(f"Patched safely: {os.path.basename(filepath)}")

print(f"Total safely patched: {patched_count}")
