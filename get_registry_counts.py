from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import neuralops.attentions, neuralops.compression_engine
import json

attn_keys = len(ATTENTION_REGISTRY.keys())
comp_keys = len(COMPRESSOR_REGISTRY.keys())
try:
    with open('neuralops/core/deprecated_aliases.json') as f:
        dep_count = len(json.load(f))
except Exception:
    dep_count = 0

print(f"ATTN_CANONICAL_KEYS: {attn_keys}")
print(f"COMP_CANONICAL_KEYS: {comp_keys}")
print(f"TOTAL_REGISTRY_KEYS: {attn_keys + comp_keys}")
print(f"DEPRECATED_ALIASES: {dep_count}")
