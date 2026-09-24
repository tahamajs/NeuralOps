import sys
from pathlib import Path

p = Path("Codes/framework/run_quality_sweep.py")
content = p.read_text(encoding="utf-8")

target = """    from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    attention_modules ="""

replacement = """    from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
    # Force dynamic module discovery to populate ATTENTION_REGISTRY and COMPRESSOR_REGISTRY
    _ = modules()
    attention_modules ="""

if "_ = modules()" not in content:
    content = content.replace(target, replacement)
    p.write_text(content, encoding="utf-8")
    print("Patched")
else:
    print("Already patched")
