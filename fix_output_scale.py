import os
import re

directories = ["Codes/framework/attentions", "Codes/framework/compression_engine"]

def patch_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all __init__ methods that have a super().__init__(...) call.
    # We want to insert self.output_scale = nn.Parameter(torch.ones(1)) right after it,
    # IF it's not already there.
    
    # We'll use a regex that matches super().__init__(...) and the following newline/indentation
    pattern = re.compile(r'(super\(\)\.__init__\([^)]*\)\s*\n(\s+))')
    
    def replacer(match):
        prefix = match.group(1)
        indent = match.group(2)
        return prefix + "if not hasattr(self, 'output_scale'):\n" + indent + "    self.output_scale = nn.Parameter(torch.ones(1))\n" + indent

    new_content, count = pattern.subn(replacer, content)

    # We also need to multiply the return of forward() by self.output_scale.
    # This is trickier. Let's just do it manually for the known ones:
    # orion, oha, optimized_hybrid_attention, hgrn2_gated_deltanet
    
    if new_content != content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        return count
    return 0

total = 0
for d in directories:
    for root, _, files in os.walk(d):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    count = patch_file(filepath)
                    if count > 0:
                        total += count
                        print(f"Patched {filepath} ({count} times)")
                except Exception as e:
                    print(f"Failed {filepath}: {e}")

print(f"Total patches: {total}")
