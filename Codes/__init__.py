"""NeuralOps: SVD Linear Attention & Model Compression Framework."""

import sys
from pathlib import Path

# Automatically ensure 'Codes' is in sys.path so both
# `import framework...` and `import Codes.framework...` work seamlessly.
_CODES_DIR = str(Path(__file__).resolve().parent)
if _CODES_DIR not in sys.path:
    sys.path.insert(0, _CODES_DIR)

__version__ = "1.0.0"
