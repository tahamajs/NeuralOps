import os

stubs = {
    'adaptive_span_inject.py': 'adaptive_span_attention',
    'adaptive_sparse_inject.py': 'adaptive_sparse_attention',
    'adasplash_inject.py': 'adasplash_attention',
    'cluster_inject.py': 'cluster_attention',
    'galerkin_inject.py': 'galerkin_attention',
    'hla_inject.py': 'hla_attention',
    'kda_inject.py': 'kda_attention'
}

for stub, base in stubs.items():
    path = f'Codes/framework/attentions/{stub}'
    content = f'''"""Deprecated stub. Re-exports base module for registry compatibility."""
from framework.attentions.{base} import *  # noqa
from framework.attentions.{base} import inject_{base.replace("_attention", "")} as inject  # noqa
'''
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Converted: {stub}")
