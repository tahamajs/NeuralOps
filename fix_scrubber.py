import re

with open('Docs/visualizer/app.js', 'r') as f:
    code = f.read()

pattern = re.compile(r"const stages = \[\s*'input_embeddings',\s*'q_proj',\s*'attn\.residual_add',\s*'gate_proj',\s*'ffn\.residual_add',\s*'output_head'\s*\];", re.DOTALL)

replacement = """const stages = [
                'layer.0',
                'layer.0',
                'q_lin',
                'out_lin',
                'lin1',
                'lin2',
                'layer.11'
            ];"""

if pattern.search(code):
    code = pattern.sub(replacement, code)
    with open('Docs/visualizer/app.js', 'w') as f:
        f.write(code)
    print("Fixed scrubber stages")
else:
    print("Could not find scrubber stages array")

