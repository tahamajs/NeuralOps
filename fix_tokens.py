import sys, os
sys.path.insert(0, 'Codes/framework')
import re

with open('Codes/framework/attentions/token_compressed.py', 'r') as f:
    text = f.read()

# Fix __init__ args to match standard: total_layers instead of num_layers
text = text.replace('num_layers: int,', 'total_layers: int,')
text = text.replace('num_layers, old_attn', 'total_layers, old_attn')
text = text.replace('num_layers: int', 'total_layers: int')

with open('Codes/framework/attentions/token_compressed.py', 'w') as f:
    f.write(text)

print("token_compressed.py fixed")
