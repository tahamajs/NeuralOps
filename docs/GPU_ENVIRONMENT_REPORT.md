# GPU Environment Report

## neuralops
- Path: D:\SVD_Project\neuralops\__init__.py (Target: D:\SVD_linear_Attention\neuralops\__init__.py)
- Version: 1.2.0
- Attention: 159
- Compressor: 73
- Total: 232

## Hardware
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- VRAM: 8.59 GB
- CUDA: 12.4 (Driver 566.07, CUDA 12.7 support)

## Python
- Python: 3.13.5 (Anaconda3)
- PyTorch: 2.6.0+cu124
- Triton: 3.4.0 (HAS_TRITON: True)

## CLI
- `python -m neuralops.benchmarks --help`: works
- `python -m neuralops.benchmarks list`: works
- `python -m neuralops.benchmarks verify`: reports 159 attention + 73 compressor modules
