# neuralops Unified Entry Point

## Single Command

    python -m neuralops <command>

## Available Commands

| Command | Description | Example |
|---|---|---|
| `info` | Show project info, hardware, and registry size | `python -m neuralops info` |
| `list` | List registered modules | `python -m neuralops list --type attention` |
| `bench` | Run benchmark | `python -m neuralops bench --module svd_linear_attention` |
| `train` | Train model | `python -m neuralops train --config configs/x.yaml` |
| `compress` | Compress model | `python -m neuralops compress --model gpt2 --method svd` |
| `eval` | Evaluate model | `python -m neuralops eval --model gpt2` |
| `test` | Run tests | `python -m neuralops test` |
| `serve` | Start API server | `python -m neuralops serve --port 8000` |
| `doctor` | Diagnose installation, PyTorch, CUDA, and registry | `python -m neuralops doctor` |

## Reachability Guarantees

All modules across the entire repository are reachable via at least one of:
  1. `python -m neuralops <command>`
  2. `neuralops.core.registry.ATTENTION_REGISTRY.get(name)`
  3. `from neuralops.<subpackage> import <Symbol>`

Zero orphan modules exist in `neuralops`.

## Adding a New Command

1. Create `neuralops/cli/<name>_cmd.py` with `cmd_<name>(args)`
2. Register the subcommand in `neuralops/__main__.py`
3. Verify with `python -m neuralops <name> --help`
