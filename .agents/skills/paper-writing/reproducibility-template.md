# Reproducibility Statement Template

## Sections

1. **Environment setup** — git clone, pip install, exact versions
2. **Running experiments** — copy-pasteable commands
3. **Downloading checkpoints** — huggingface_hub or curl
4. **Reproducing individual tables** — command per table
5. **Random seeds** — all seeds used
6. **Expected results** — accuracy ± tolerance
7. **Data availability** — links to datasets and checkpoints
8. **Known issues** — encoding, CUDA version, batch size

## Rules

- Every claim in the paper must be reproducible from a single command
- Seed should be documented for every run
- Expected results must have tolerances
