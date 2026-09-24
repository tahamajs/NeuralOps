import glob
import os
import json
import re

files = glob.glob('Codes/framework/attentions/*.py') + glob.glob('Codes/framework/compression_engine/*.py') + glob.glob('Codes/framework/*.py')
mapping = {}

prompt_names = ['rala.py', 'svd_linear.py', 'compressed_mha.py', 'compressed_attention_layer.py', 'gqa.py', 'jamba2.py', 'aegis_v4_module.py', 'aegis_v5_module.py', 'aasvd_compressor.py', 'simple_ssm.py', 'routing.py', 'run_agnews_multi_class_sweep.py']

for name in prompt_names:
    print(f"Resolving: {name}")

for f in files:
    content = open(f, 'r', encoding='utf-8').read()
    b = os.path.basename(f)
    if 'alpha.requires_grad' in content and 'rala' in b: mapping['rala.py'] = b
    if '_is_target_module' in content and 'svd_linear' in b: mapping['svd_linear.py'] = b
    if 'copy_weights' in content and 'compressed_mha' in b: mapping['compressed_mha.py'] = b
    if 'output_scale' in content and 'compressed_attention' in b: mapping['compressed_attention_layer.py'] = b
    if 'num_kv_heads' in content and 'gqa' in b: mapping['gqa.py'] = b
    if 'SVDMambaSub' in content and 'jamba' in b: mapping['jamba2.py'] = b
    if 'softpick' in content and 'aegis_v4' in b: mapping['aegis_v4_module.py'] = b
    if 'output_scale' in content and 'aegis_v5' in b: mapping['aegis_v5_module.py'] = b
    if 'max_error' in content and 'aa_svd' in b: mapping['aasvd_compressor.py'] = b
    if 'out_proj' in content and 'ssm' in b: mapping['simple_ssm.py'] = b
    if 'kmeans_assign' in content: mapping['routing.py'] = b
    if 'run_agnews_multi_class_sweep' in b: mapping['run_agnews_multi_class_sweep.py'] = b

print(json.dumps(mapping, indent=2))
with open('papers/_rank_ablation/filename_map.json', 'w') as f:
    json.dump(mapping, f, indent=2)
