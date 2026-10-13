# GTX 1060 6GB — Environment Setup

## Hardware
- CPU: Intel i7-7700HQ (4C/8T, Kaby Lake)
- RAM: 14 GB
- GPU: NVIDIA GeForce GTX 1060 Mobile 6GB (Pascal, compute 6.1)
- Driver: 580.178.04
- CUDA Runtime: 13.0 (backward compatible)
- Disk: 256GB SSD (root), 1TB HDD (/dev/sdb, unmounted)

## Software
- Python: 3.12.15 (from deadsnakes PPA)
- PyTorch: 2.5.1+cu124
- Torchvision: 0.20.1
- Torchaudio: 2.5.1
- Transformers: 4.46.3
- Datasets: 3.1.0
- Triton: 3.1.0

## Known Limitations
- NO flash-attn (compute < 7.5)
- NO native bf16 tensor core acceleration (Pascal limitation)
- NO FP8 (Pascal limitation)
- FP16 works but no tensor cores -> slower than modern RTX GPUs
- Max batch for BERT-base: 8 (with grad accumulation)
- Max batch for BERT-large: 2 (with gradient checkpointing)

## Recommended Env Vars
export TORCH_CUDA_ARCH_LIST="6.1"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

## Usage
cd ~/Project
source .venv/bin/activate
python your_script.py

## Jupyter
jupyter lab --no-browser --port=8888
Kernel: 'Python 3.12 (SVD Project)'
