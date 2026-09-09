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
        
        replacements = [
            (r"return output, None", r"return output * self.output_scale, None"),
            (r"return output\n", r"return output * self.output_scale\n"),
            (r"return output\r\n", r"return output * self.output_scale\r\n"),
            (r"return out, None", r"return out * self.output_scale, None"),
            (r"return out\n", r"return out * self.output_scale\n"),
            (r"return out\r\n", r"return out * self.output_scale\r\n"),
            (r"return context, None", r"return context * self.output_scale, None"),
            (r"return context\n", r"return context * self.output_scale\n"),
            (r"return context\r\n", r"return context * self.output_scale\r\n"),
            (r"return ctx, None", r"return ctx * self.output_scale, None"),
            (r"return ctx\n", r"return ctx * self.output_scale\n"),
            (r"return ctx\r\n", r"return ctx * self.output_scale\r\n"),
        ]
        
        # Only patch if self.output_scale is in __init__
        if "self.output_scale" in content:
            # Check if already patched to avoid double patching
            if "output * self.output_scale" not in content and "out * self.output_scale" not in content and "context * self.output_scale" not in content and "ctx * self.output_scale" not in content:
                for old, new in replacements:
                    content = re.sub(old, new, content)

        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            patched_count += 1
            print(f"Patched forward directly: {os.path.basename(filepath)}")

print(f"Total safely patched: {patched_count}")
