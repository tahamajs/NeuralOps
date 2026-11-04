import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from neuralops.benchmarks.adapters.cuda_adapter import CUDAKernelAdapter, CANONICAL_CUDA_MODULES
from neuralops.benchmarks.adapters.triton_adapter import TritonKernelAdapter, CANONICAL_TRITON_MODULES, EXTENSION_TRITON_MODULES

print("All imports succeeded!")
