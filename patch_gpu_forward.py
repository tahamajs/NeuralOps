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
        
        # Check if we already injected output_scale in __init__
        if "self.output_scale = nn.Parameter" in content:
            # We need to find the forward function
            # and replace the final return to multiply by self.output_scale
            # This is tricky because `return out, None` vs `return out`
            # Let's just find `def forward` and inside it, replace `return ` with multiplying the first return element
            
            # Since this is a bit too risky with regex across arbitrary files,
            # Let's do something simpler:
            # We can look for `return output, None` -> `return output * self.output_scale, None`
            # `return output` -> `return output * self.output_scale`
            # `return out, None` -> `return out * self.output_scale, None`
            # `return out` -> `return out * self.output_scale`
            # `return context` -> `return context * self.output_scale`
            # `return ctx` -> `return ctx * self.output_scale`
            
            # Let's ensure we only replace inside `forward`
            parts = content.split('def forward(')
            if len(parts) > 1:
                forward_part = parts[1]
                
                # Replace common returns
                replacements = [
                    (r"return output, None", r"return output * self.output_scale, None"),
                    (r"return output\n", r"return output * self.output_scale\n"),
                    (r"return out, None", r"return out * self.output_scale, None"),
                    (r"return out\n", r"return out * self.output_scale\n"),
                    (r"return context, None", r"return context * self.output_scale, None"),
                    (r"return context\n", r"return context * self.output_scale\n"),
                    (r"return ctx, None", r"return ctx * self.output_scale, None"),
                    (r"return ctx\n", r"return ctx * self.output_scale\n"),
                ]
                
                for old, new in replacements:
                    forward_part = re.sub(old, new, forward_part)
                
                content = parts[0] + 'def forward(' + forward_part

        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            patched_count += 1
            print(f"Patched forward in: {os.path.basename(filepath)}")

print(f"Total files patched: {patched_count}")
