import os
import re

files = ['rpc_attention', 'svd_llm_v2_attention', 'dobi_attention', 'nystrom_attention', 'tt', 'hgrn2_layer', 'SOAFusedAttention', 'adaptive_svd_attention', 'decay_linear_attention', 'routing_attention', 'salt', 'simple_ssm', 'svd_projection_attention', 'template_attention', 'token_compressed_attention']

for m in files:
    p = f'thesis_final/attentions_latex/individual_modules/{m}.tex'
    with open(p, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.startswith(r'\section{') and line.strip().endswith('}}}'):
            # Replace ending }}} with }}
            line = line.rstrip()[:-1] + '\n'
        elif line.startswith(r'\label{') and line.strip().endswith('}}'):
            # Replace ending }} with }
            line = line.rstrip()[:-1] + '\n'
        new_lines.append(line)
        
    with open(p, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

print('Successfully cleaned header braces.')
