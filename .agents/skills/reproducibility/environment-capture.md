# Environment Capture

## Commands

```bash
python -c "import torch; print(torch.__version__)" > env/pytorch.txt
python -c "import torch; print(torch.version.cuda)" > env/cuda.txt
pip freeze > env/requirements.txt
git rev-parse HEAD > env/git.txt
nvidia-smi > env/hardware.txt
python -c "import platform; print(platform.platform())" > env/os.txt
```

## Output

- `papers/modules/<module>/env/pytorch.txt`
- `papers/modules/<module>/env/cuda.txt`
- `papers/modules/<module>/env/requirements.txt`
- `papers/modules/<module>/env/git.txt`
- `papers/modules/<module>/env/hardware.txt`
- `papers/modules/<module>/env/os.txt`
