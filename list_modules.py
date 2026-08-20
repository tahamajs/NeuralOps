import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from main import _import_framework
fw = _import_framework()
attns = list(fw["ATTENTION_REGISTRY"].keys())
comps = list(fw["COMPRESSOR_REGISTRY"].keys())
print(f"Total Attentions: {len(attns)}")
print(f"Total Compressors: {len(comps)}")
print("Sample Attentions:", attns[:25])
print("Sample Compressors:", comps[:25])
