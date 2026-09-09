import os, re
path = 'D:/SVD_Project/Codes/framework/compression_engine/a3_compressor.py'
with open(path, 'r', encoding='utf-8') as file:
    content = file.read()
    
# Insert output_scale
content = re.sub(r'(super\(\)\.__init__\([^)]*\)\s*\n(\s+))', r"\1if not hasattr(self, 'output_scale'):\n\2    import torch\n\2    import torch.nn as nn\n\2    self.output_scale = nn.Parameter(torch.ones(1))\n\2", content)
content = content.replace('return out', 'return out * self.output_scale')

with open(path, 'w', encoding='utf-8') as file:
    file.write(content)
print('Patched a3')
