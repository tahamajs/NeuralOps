# NeuralOps baseline — 2026-11-06 (commit 5e84792c670461277b8687716d63b848fec34151)

> **Environment Note**: Per user instruction, the existing project virtual environment at `/Users/tahamajs/dev/Project/.venv` is authoritative for this baseline and all future tasks. The temporary `/tmp/neuralops-baseline-venv` was abandoned per user instruction. Where a step was previously run in the `/tmp` environment, results for both environments are recorded below.

## Environment
- **Active / Authoritative Virtualenv**: `/Users/tahamajs/dev/Project/.venv`
  - **Python**: `Python 3.14.3` (`/Users/tahamajs/dev/Project/.venv/bin/python`)
  - **pip**: `pip 25.3 from /Users/tahamajs/dev/Project/.venv/lib/python3.14/site-packages/pip (python 3.14)`
  - **pytest**: `pytest 9.1.1` (`/Users/tahamajs/dev/Project/.venv/bin/pytest`)
  - **ruff**: `ruff 0.16.10` (`/Users/tahamajs/dev/Project/.venv/bin/ruff`)
- **System Environment**:
  - **Python (system)**: `Python 3.14.3` (`/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`)
  - **pip (system)**: `pip 26.2.1 from /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/pip (python 3.14)`
  - **macOS version (`sw_vers`)**:
    ```text
    ProductName:		macOS
    ProductVersion:		27.2
    BuildVersion:		26B5112a
    ```
  - **CPU architecture (`uname -m`)**: `arm64`
- **Abandoned venv (`/tmp/neuralops-baseline-venv`)**:
  - Python 3.14.3, pip upgraded to 26.3, abandoned per user directive.

## Git state
- **Branch**: `main`
- **HEAD commit**: `5e84792c670461277b8687716d63b848fec34151`
- **Recent commit log (`git log --oneline -10`)**:
  ```text
  5e84792c67 (HEAD -> main, origin/main, origin/HEAD) docs(ui): finalize Web UI architecture guide and end-to-end integration tests
  eb7caae2bd feat(ui): add static site generator for HuggingFace Spaces distribution
  dea1f0c337 feat(ui): add Cloudflare tunnel share mode and HTTP Basic Auth middleware
  e3880d4e37 feat(ui): add Model Zoo and Notebook demonstration catalog
  2977df8dfc feat(ui): add Observable Plot roofline and pareto charts to benchmarks explorer
  f1bc7e9ca1 feat(benchmarks): add BaseBenchmarkAdapter with 3-gate correctness lifecycle
  d20bb09898 feat(benchmarks): wire canonical benchmark flag and fail-closed checks into CLI
  77d5144416 feat(ui): add HTMX dynamic partial endpoints and live polling
  10abb60e73 feat(ui): implement core pages and routes for dashboard, attentions, compressors, benchmarks, and doctor
  e05d5bfcb5 feat(benchmarks): implement canonical status enum and unified benchmark registry
  ```
- **Git status before task (`git status --short`)**:
  ```text
  ?? .agents/agt-prompts/
  ```
- **Git status with BASELINE.md (`git status --short`)**:
  ```text
  ?? .agents/agt-prompts/
  ?? BASELINE.md
  ```

## Editable install
### In Authoritative Project Environment (`/Users/tahamajs/dev/Project/.venv`)
- **Command**: `/Users/tahamajs/dev/Project/.venv/bin/python -m pip install -e .`
- **Exit code**: `0`
- **Output summary**: Built `neuralops-2.0.6-0.editable-py3-none-any.whl`, cleanly uninstalled existing `neuralops-2.0.5`, and installed `neuralops-2.0.6`.
- **Last 30 lines of installation output**:
  ```text
  Requirement already satisfied: mdurl~=0.1 in ./.venv/lib/python3.14/site-packages (from markdown-it-py>=2.2.0->rich>=13.0->neuralops==2.0.6) (0.1.2)
  Requirement already satisfied: numpy>=1.24.1 in ./.venv/lib/python3.14/site-packages (from scikit-learn>=1.3->neuralops==2.0.6) (2.4.4)
  Requirement already satisfied: scipy>=1.10.0 in ./.venv/lib/python3.14/site-packages (from scikit-learn>=1.3->neuralops==2.0.6) (1.17.1)
  Requirement already satisfied: joblib>=1.3.0 in ./.venv/lib/python3.14/site-packages (from scikit-learn>=1.3->neuralops==2.0.6) (1.5.3)
  Requirement already satisfied: threadpoolctl>=3.2.0 in ./.venv/lib/python3.14/site-packages (from scikit-learn>=1.3->neuralops==2.0.6) (3.6.0)
  Requirement already satisfied: shellingham>=1.3.0 in ./.venv/lib/python3.14/site-packages (from typer>=0.20.0->huggingface-hub>=0.20->neuralops==2.0.6) (1.5.4)
  Requirement already satisfied: annotated-doc>=0.0.2 in ./.venv/lib/python3.14/site-packages (from typer>=0.20.0->huggingface-hub>=0.20->neuralops==2.0.6) (0.0.4)
  Requirement already satisfied: sniffio>=1.1 in ./.venv/lib/python3.14/site-packages (from anyio->httpx<1,>=0.23.0->huggingface-hub>=0.20->neuralops==2.0.6) (1.3.1)
  Building wheels for collected packages: neuralops
    Building editable for neuralops (pyproject.toml) ... -   \   |   /   -   \   |   /   -   \   done
    Created wheel for neuralops: filename=neuralops-2.0.6-0.editable-py3-none-any.whl size=73044 sha256=d2aaa288b04ca8a8b8df611668cf868fa5c6e5c7190a05221f2ed02a9da8199d
    Stored in directory: /private/var/folders/cg/l2rdx46d6lv3b5xc17b420yc0000gn/T/pip-ephem-wheel-cache-ketfijns/wheels/8d/b1/d0/1342d20c8003e76cb0288e0afd5093c457c078f63400b130b3
  Successfully built neuralops
  Installing collected packages: neuralops
    Attempting uninstall: neuralops
      Found existing installation: neuralops 2.0.5
      Uninstalling neuralops-2.0.5:
        Successfully uninstalled neuralops-2.0.5
  Successfully installed neuralops-2.0.6
  ```
- **Pip list excerpt (`pip list | grep -i -E "torch|transformers|triton|pytest|ruff|neuralops"`)**:
  ```text
  lion-pytorch                             0.2.4
  neuralops                                2.0.6        /Users/tahamajs/dev/Project
  pytest                                   9.1.1
  pytest-cov                               7.1.0
  pytest-django                            4.12.0
  pytest-xdist                             3.8.0
  ruff                                     0.16.10
  sentence-transformers                    5.5.1
  torch                                    2.11.0
  transformers                             5.7.0
  ```
  *(Note: `triton` is absent on macOS arm64 by design).*

### In Abandoned Environment (`/tmp/neuralops-baseline-venv`)
- **Command**: `/tmp/neuralops-baseline-venv/bin/python -m pip install -e .`
- **Exit code**: `0`
- **Pip list excerpt**: `neuralops 2.0.6` (torch, transformers, triton, pytest, ruff missing).

## Import check
### In Authoritative Project Environment (`/Users/tahamajs/dev/Project/.venv`)
- **Command**: `/Users/tahamajs/dev/Project/.venv/bin/python -c "import neuralops; print(neuralops.__version__)"`
- **Exit code**: `0`
- **Output**: `2.0.6`

### In Abandoned Environment (`/tmp/neuralops-baseline-venv`)
- **Command**: `/tmp/neuralops-baseline-venv/bin/python -c "import neuralops; print(neuralops.__version__)"`
- **Exit code**: `0`
- **Output**: `2.0.6`

## CLI smoke
### In Authoritative Project Environment (`/Users/tahamajs/dev/Project/.venv`)
| Command | Exit code | Result |
| :--- | :--- | :--- |
| `neuralops --help` | `0` | Works: Displays NeuralOps Engine CLI v2.0.6 banner and subcommands |
| `neuralops info` | `0` | Works: Displays system overview, PyTorch 2.11.0, and 131 attentions, 72 compressors, 26 Triton kernel modules (62 ops), 114 CUDA kernel implementations |
| `python -m neuralops --help` | `0` | Works: Displays NeuralOps Engine CLI v2.0.6 banner and subcommands |

#### Output Details in `.venv`:
1. `/Users/tahamajs/dev/Project/.venv/bin/neuralops --help` (exit 0):
```text

  _   _                      _  ____
 | \ | | ___ _   _ _ __ __ _| |/ __ \ _ __  ___
 |  \| |/ _ \ | | | '__/ _` | | |  | | '_ \/ __|
 | |\  |  __/ |_| | | | (_| | | |__| | |_) \__ \
 |_| \_|\___|\__,_|_|  \__,_|_|\____/| .__/|___/
                                     |_|

╭──────────────────────────── ⚡ NeuralOps Engine ─────────────────────────────╮
│                                                                              │
│  NeuralOps v2.0.6 — High-Performance Attention Zoo, SVD Compression & Fused  │
│  Triton Kernels                                                              │
│  https://github.com/tahamajs/SVD_linear_Attention                            │
│                                                                              │
╰─────────────── Pair Programming & Distributed ML Systems CLI ────────────────╯

Usage: neuralops [OPTIONS] <command> [ARGS]...

                    Model Surgery, Composition & Compression                    
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Command              ┃ Description                                           ┃
```

2. `/Users/tahamajs/dev/Project/.venv/bin/neuralops info` (exit 0):
```text
╭───────────────────────── NeuralOps System Overview ──────────────────────────╮
│ NeuralOps Deep Learning & High-Throughput Attention Suite                    │
│ Framework Version: 2.0.6                                                     │
│ Universal Sub-Quadratic Attention, Continuous Batching, and SVD Compression  │
╰──────────────────────────────────────────────────────────────────────────────╯
                   Hardware & Runtime Environment                   
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property                  ┃ Value                                ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Python Version            │ 3.14.3                               │
│ PyTorch Version           │ 2.11.0                               │
│ Triton JIT Version        │ Not installed (PyTorch eager active) │
│ CUDA Acceleration         │ Disabled (CPU only)                  │
│ Host OS / Platform        │ Darwin 27.2.0 (arm64)                │
└───────────────────────────┴──────────────────────────────────────┘
                                Canonical Component Inventory                   
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━
┃ Operator Category              ┃  Canonical Count   ┃ Mathematical Architectur
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━
│ Attention Mechanisms           │    131 modules     │ 11 mathematical families
│                                │                    │ Linear, GQA, SSM, TTT, S
│ Compression Engines            │     72 engines     │ 7 algorithmic families (
│                                │                    │ Quantization, Pruning, K
│ Triton Kernel Modules          │   26 modules (62   │ Specialized JIT fusions 
│                                │ unified operators) │ kernels across 4 tiers) 
│ CUDA Kernel Implementations    │   114 operators    │ Accelerated C++/CUDA ker
│                                │                    │ extreme throughput      
└────────────────────────────────┴────────────────────┴─────────────────────────
Status: Production-grade runtime ready. Run neuralops list to view complete 
module inventory.
```

3. `/Users/tahamajs/dev/Project/.venv/bin/python -m neuralops --help` (exit 0):
```text

  _   _                      _  ____
 | \ | | ___ _   _ _ __ __ _| |/ __ \ _ __  ___
 |  \| |/ _ \ | | | '__/ _` | | |  | | '_ \/ __|
 | |\  |  __/ |_| | | | (_| | | |__| | |_) \__ \
 |_| \_|\___|\__,_|_|  \__,_|_|\____/| .__/|___/
                                     |_|

╭──────────────────────────── ⚡ NeuralOps Engine ─────────────────────────────╮
│                                                                              │
│  NeuralOps v2.0.6 — High-Performance Attention Zoo, SVD Compression & Fused  │
│  Triton Kernels                                                              │
│  https://github.com/tahamajs/SVD_linear_Attention                            │
│                                                                              │
╰─────────────── Pair Programming & Distributed ML Systems CLI ────────────────╯

Usage: neuralops [OPTIONS] <command> [ARGS]...

                    Model Surgery, Composition & Compression                    
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Command              ┃ Description                                           ┃
```

### In Abandoned Environment (`/tmp/neuralops-baseline-venv`)
| Command | Exit code | Result |
| :--- | :--- | :--- |
| `neuralops --help` | `0` | Works: banner and subcommands displayed |
| `neuralops info` | `1` | Fails: `ModuleNotFoundError: No module named 'torch'` |
| `python -m neuralops --help` | `0` | Works: banner and subcommands displayed |

## Pytest collection
### In Authoritative Project Environment (`/Users/tahamajs/dev/Project/.venv`)
- **Command**: `/Users/tahamajs/dev/Project/.venv/bin/python -m pytest --collect-only -q`
- **Exit code**: `2`
- **Collected count**: `7778 tests collected`
- **Collection errors**: `3 errors in 43.14s`
- **Files that failed collection**:
  1. `tests/test_v135_flagship_subsystems.py`
     - Error: `DeprecationWarning: neuralops.core.roofline has moved to neuralops.analysis.roofline and will be deprecated in v1.7.0` (raised as error due to pytest.ini `filterwarnings = error`).
  2. `tests/test_v136_extensibility_and_integration.py`
     - Error: `DeprecationWarning: neuralops.core.roofline has moved to neuralops.analysis.roofline and will be deprecated in v1.7.0` (raised as error due to pytest.ini `filterwarnings = error`).
  3. `tests/test_v17_ergonomics.py`
     - Error: `ImportError: cannot import name 'pipeline' from 'neuralops' (/Users/tahamajs/dev/Project/neuralops/__init__.py)`.

### In Abandoned / System Environments
- `/tmp/neuralops-baseline-venv`: Exit code `1` (`No module named pytest`).
- System Python (`python3 -m pytest --collect-only -q`): Exit code `134` (`SIGABRT`, C stack recursion abort in CPython `_Py_CheckRecursiveCall`).

## Ruff
### In Authoritative Project Environment (`/Users/tahamajs/dev/Project/.venv`)
- **Command**: `/Users/tahamajs/dev/Project/.venv/bin/ruff check .`
- **Exit code**: `1`
- **Total error count**: `Found 2 errors.`
- **Reported findings**:
  ```text
  F404 `from __future__` imports must occur at the beginning of the file
    --> neuralops/benchmarks/legacy_root/benchmark_dispatch_integrity.py:14:1
     |
  12 | warnings.warn("Deprecated. Use neuralops benchmark instead.", DeprecationWarning, stacklevel=2)
  13 |
  14 | from __future__ import annotations
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  15 |
  16 | import sys
     |

  F404 `from __future__` imports must occur at the beginning of the file
    --> neuralops/benchmarks/legacy_root/benchmark_engine_matrix.py:14:1
     |
  12 | warnings.warn("Deprecated. Use neuralops benchmark instead.", DeprecationWarning, stacklevel=2)
  13 |
  14 | from __future__ import annotations
     | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  15 |
  16 | import sys
     |

  Found 2 errors.
  ```

### In Abandoned / System Environments
- `/tmp/neuralops-baseline-venv/bin/python -m ruff check .`: Exit code `1` (`No module named ruff`).
- Direct shell `ruff check .` (without PATH): Exit code `127` (`command not found: ruff`).
- System `python3 -m ruff check .`: Exit code `1`, `Found 2 errors.` (identical to `.venv`).

## Commands that already fail
Every command listed below exited non-zero in the authoritative `.venv` baseline (with historical notes from initial runs). This list serves as the canonical pre-existing failure baseline: a failure matching one of these in subsequent tasks is an existing condition, not a newly introduced regression.

1. **`python -m pytest --collect-only -q` (in `.venv`)**
   - **Exit code**: `2`
   - **Summary**: `7778 tests collected, 3 errors in 43.14s`.
   - **Failures**:
     - `tests/test_v135_flagship_subsystems.py`: `DeprecationWarning: neuralops.core.roofline has moved to neuralops.analysis.roofline and will be deprecated in v1.7.0` (treated as error under strict warnings filter).
     - `tests/test_v136_extensibility_and_integration.py`: `DeprecationWarning: neuralops.core.roofline has moved to neuralops.analysis.roofline and will be deprecated in v1.7.0` (treated as error under strict warnings filter).
     - `tests/test_v17_ergonomics.py`: `ImportError: cannot import name 'pipeline' from 'neuralops'`.

2. **`ruff check .` (in `.venv`)**
   - **Exit code**: `1`
   - **Summary**: `Found 2 errors.`
   - **Failures**:
     - `neuralops/benchmarks/legacy_root/benchmark_dispatch_integrity.py:14:1`: `F404 from __future__ imports must occur at the beginning of the file`
     - `neuralops/benchmarks/legacy_root/benchmark_engine_matrix.py:14:1`: `F404 from __future__ imports must occur at the beginning of the file`

3. **Historical / Non-.venv Failures**:
   - `neuralops info` in clean minimal environment without `torch`: Exit code `1` (`ModuleNotFoundError: No module named 'torch'`).
   - `python3 -m pytest --collect-only -q` in unconfigured system Python: Exit code `134` (`SIGABRT`, C-stack recursion abort under Python 3.14.3).
   - Shell command `ruff` outside `.venv`: Exit code `127` (`command not found`).
