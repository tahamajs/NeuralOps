# Reproducibility

## Environment

All experiments are reproducible with:
- Python 3.10+
- PyTorch 2.2.0+
- CUDA 12.x (for GPU runs)

## Seeds

All benchmarks use `torch.manual_seed(42)` by default. Override with `--seed`.

## Hardware

Reference results are from:
- GPU: NVIDIA A100 / H100
- CPU: Apple M-series / x86_64

## Running Benchmarks

```bash
python -m neuralops.benchmarks run \
    --module <module_name> \
    --model <model_name> \
    --dataset <dataset> \
    --rank 64 \
    --seed 42
```
