import sys
from pathlib import Path
cwd = Path.cwd()
if str(cwd / "Codes") not in sys.path:
    sys.path.insert(0, str(cwd / "Codes"))
import framework.attentions
import framework.compression_engine
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

print("ATTENTIONS:", list(ATTENTION_REGISTRY.keys()))
print("COMPRESSORS:", list(COMPRESSOR_REGISTRY.keys()))
