import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CODES = HERE.parent / "Codes"
ROOT = HERE.parent
for p in [str(CODES), str(HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import framework.attentions
import framework.compression_engine
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

pairs = [
    ('cosformer', 'cosformer_inject'),
    ('gqa', 'gqa_attention'),
    ('mamba', 'mamba_inject'),
    ('soa_attn', 'soa_module'),
    ('soa_attn', 'soa_fused_module'),
    ('leka_v1', 'leka_v1_inject'),
    ('leka_v2', 'leka_v2_inject'),
    ('leka_v3', 'leka_v3_inject'),
    ('replace', 'replace_attention'),
    ('svd_attention', 'svd_projection_attention'),
    ('kda_attention', 'kda_inject'),
    ('galerkin', 'galerkin_attention'),
]

print("=" * 80)
print("ALIAS PAIR VERIFICATION REPORT")
print("=" * 80)

for a, b in pairs:
    ca = ATTENTION_REGISTRY.get(a) or COMPRESSOR_REGISTRY.get(a)
    cb = ATTENTION_REGISTRY.get(b) or COMPRESSOR_REGISTRY.get(b)
    is_same = (ca is cb) and (ca is not None)
    name_a = getattr(ca, '__name__', str(ca)) if ca else 'None'
    name_b = getattr(cb, '__name__', str(cb)) if cb else 'None'
    type_a = type(ca).__name__ if ca else 'None'
    type_b = type(cb).__name__ if cb else 'None'
    print(f"{a:25s} vs {b:25s} -> is_same: {str(is_same):5s} | {name_a} ({type_a}) vs {name_b} ({type_b})")
