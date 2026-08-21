import re

with open("Docs/app.js", "r") as f:
    content = f.read()

new_map = """            const blogMap = {
                '01': ['svd_linear', 'aegis', 'svd', 'asvd', 'nystrom', 'spectra_v2', 'full'],
                '02': ['mamba', 'gated_delta_net', 'linformer', 'performer', 'flash3', 'griffin_ssd', 'samba_m3'],
                '03': ['mla', 'nsa', 'adaptive', 'sparse', 'adasplash'],
                '04': ['lora', 'lowrank', 'svd_compressed', 'lora_attn'],
                '05': ['quant', 'prun', 'fp8', 'int4']
            };"""

pattern = re.compile(r'            const blogMap = \{.*?\};', re.DOTALL)
new_content = pattern.sub(new_map, content)

with open("Docs/app.js", "w") as f:
    f.write(new_content)
