# Sharif University HPC Cluster & RTX 3090 GPU Master Execution Guide

> **Authoritative Operational Reference for NeuralOps AI Coding Agents and Human Engineers.**
> Grounded in production-verified deployment on the **Sharif University HPC Center** (`hpc.itc.sharif.edu`).

---

## 1. Executive Hardware & Environment Profile

| Property | Value / Specification | Notes |
| :--- | :--- | :--- |
| **Cluster Hostname** | `hpc.itc.sharif.edu` | Login node: `login-3` |
| **Compute Node** | `GPU-3090-1` (`10.0.0.x` internal subnet) | Direct SSH accessible from `login-3` without password |
| **Primary GPU** | **NVIDIA GeForce RTX 3090** (GA102 Ampere) | 24,268 MiB (24 GB) GDDR6X VRAM, 350W TDP |
| **CUDA Cores / SMs** | 10,496 CUDA Cores, 328 Tensor Cores (82 SMs) | Compute Capability **8.6** |
| **Host OS** | Rocks 6.1 / CentOS Linux 6.9 | Linux Kernel `2.6.32-696.el6.x86_64` |
| **Installed GPU Driver** | **NVIDIA Driver 465.19.01** | Max supported CUDA version: **CUDA 11.3** (`driver_version = 11030`) |
| **Torque PBS Queue** | **`N3090`** | Active, verified queue routing directly to `GPU-3090-1` |
| **Required Linux Group** | **`HPC` (GID 2423)** | User `tahamajlesi.ut` is verified in `/etc/group` |

---

## 2. Zero-Drop SSH Multiplexing (`ControlMaster`)

To prevent connection drops and eliminate the 5–15 second TCP handshake latency over slow internet connections, the local Mac `~/.ssh/config` **MUST** include `ControlMaster`:

```sshconfig
Host hpc
    HostName hpc.itc.sharif.edu
    User tahamajlesi.ut
    Port 22
    HostKeyAlgorithms +ssh-rsa
    PubkeyAcceptedKeyTypes +ssh-rsa
    IdentityFile ~/.ssh/id_rsa_hpc
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 5
    ConnectTimeout 15
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 1h
    WarnWeakCrypto no
    LogLevel ERROR
    StrictHostKeyChecking accept-new
```

### Verification
Run once to establish master socket:
```bash
ssh hpc "date"
```
Subsequent commands will execute instantly in under 1 second.

---

## 3. The Golden Software Stack (PyTorch 1.12.1 + CUDA 11.3)

### ⚠️ The Critical Incompatibility Trap
- The host GPU driver is `465.19.01`, which provides CUDA runtime `11.3` (`11030`).
- Modern PyTorch (`torch>=2.0`) requires NVIDIA Driver `>=525.60` (CUDA 12.x). Attempting to run PyTorch 2.x on `GPU-3090-1` results in:
  `CUDA initialization: The NVIDIA driver on your system is too old (found version 11030)`
- **Solution:** NeuralOps runs PyTorch `1.12.1+cu113` paired with transparent **backward-compatibility polyfills** (`torch.nn.functional.scaled_dot_product_attention`, `torch.compile`, `torch.amp`, `torch.nn.RMSNorm`).

### Environment Paths & Export Contract
Every command or script running on `GPU-3090-1` **MUST** export these exact paths:

```bash
# 1. UTF-8 Encoding (prevents charmap UnicodeEncodeError on CentOS 6)
export PYTHONIOENCODING=utf-8

# 2. PYTHONPATH (links pre-compiled PyTorch 1.12.1 site-packages + NeuralOps repo)
export PYTHONPATH="/home/tahamajlesi.ut/venv_cu113/lib/python3.10/site-packages:/home/tahamajlesi.ut/NeuralOps:$PYTHONPATH"

# 3. Dynamic Shared Libraries (CUDA 11.3, Intel MKL, GCC 8.2 runtime)
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:/share/apps/Anaconda/anaconda3.8/lib:/share/apps/gcc/gcc-8.2.0/lib64:/home/tahamajlesi.ut/venv_cu113/lib:$LD_LIBRARY_PATH"

# 4. Offline HuggingFace (prevents DNS timeouts on restricted cluster nodes)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 5. Python Executable
PYTHON_EXEC="/home/tahamajlesi.ut/.conda/envs/neuralops/bin/python"
```

---

## 4. Operational Rules for AI Agents (Never Waste GPU Time)

1. **Never attempt local CUDA compilation on macOS:**
   Native CUDA C++, Triton JIT, and PyTorch CUDA extensions do not exist on macOS Darwin. Always execute on `GPU-3090-1`.
2. **Never submit jobs to inactive queues:**
   Only queue **`N3090`** (and `SP`) is enabled by cluster administration. Submitting to `cuda7`, `cuda9`, `DP`, or `DDP` will fail with `Unauthorized Request` or queue disabled.
3. **Never request `:gpus=1` in PBS on N3090:**
   The Maui scheduler feature for `GPU-3090-1` is `[N3090]`, NOT `[gpus=1]`.
   - ❌ **Wrong:** `#PBS -l nodes=1:ppn=1:gpus=1` (Rejected with `Rejection Reasons: [Features: 21]`).
   - ✅ **Correct:** `#PBS -l nodes=1:ppn=1` and `#PBS -q N3090`.
4. **Never run long training or benchmark jobs inside fragile interactive SSH sessions:**
   SSH sessions over consumer internet will disconnect and send `SIGHUP` to child processes. Always submit jobs via **Torque PBS (`qsub`)** or run inside **`screen`** on `login-3`.
5. **Always maximize batch size and sequence length to saturate 24GB VRAM:**
   Small batch sizes leave the RTX 3090 at 10W and 0% compute utilization. Use `batch_size=32` and `seq_len=512` with mixed precision to hit **100% GPU-Util** and **340W TDP**.

---

## 5. Workload Execution Workflows

### Workflow A: Production Batch Job via Torque PBS (Recommended)
This workflow is completely disconnect-proof. The job runs under cluster daemon control and stores outputs in `results/`.

Create or use `~/NeuralOps/run_heavy_gpu_stress.pbs`:
```bash
#!/bin/bash
#PBS -N neuralops_gpu_workload
#PBS -q N3090
#PBS -l nodes=1:ppn=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /home/tahamajlesi.ut/NeuralOps/results/gpu_job_output.log

export PYTHONIOENCODING=utf-8
export PYTHONPATH="/home/tahamajlesi.ut/venv_cu113/lib/python3.10/site-packages:/home/tahamajlesi.ut/NeuralOps:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:/share/apps/Anaconda/anaconda3.8/lib:/share/apps/gcc/gcc-8.2.0/lib64:/home/tahamajlesi.ut/venv_cu113/lib:$LD_LIBRARY_PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/tahamajlesi.ut/NeuralOps
mkdir -p results/checkpoints

/home/tahamajlesi.ut/.conda/envs/neuralops/bin/python -u scripts/benchmarks/gpu_stress_heavy_trainer.py \
    --steps 3000 \
    --batch-size 32 \
    --seq-len 512 \
    --hidden-dim 1024 \
    --num-layers 12 \
    --num-heads 16 \
    --lr 3e-4
```

Submit command:
```bash
ssh hpc "qsub ~/NeuralOps/run_heavy_gpu_stress.pbs"
```

### Workflow B: Interactive Screen Session on `login-3`
If you need interactive observation:
```bash
# 1. Connect to login node and attach/create screen
ssh hpc
screen -S neuralops_gpu

# 2. Inside screen, SSH directly to compute node
ssh GPU-3090-1

# 3. Export environment
export PYTHONIOENCODING=utf-8
export PYTHONPATH="/home/tahamajlesi.ut/venv_cu113/lib/python3.10/site-packages:/home/tahamajlesi.ut/NeuralOps:$PYTHONPATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:/share/apps/Anaconda/anaconda3.8/lib:/share/apps/gcc/gcc-8.2.0/lib64:/home/tahamajlesi.ut/venv_cu113/lib:$LD_LIBRARY_PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd ~/NeuralOps

# 4. Launch your workload
/home/tahamajlesi.ut/.conda/envs/neuralops/bin/python scripts/benchmarks/gpu_master_suite.py --category all --device cuda --export
```
Detach anytime with `Ctrl+A, D`. Reattach later with `screen -r neuralops_gpu`.

---

## 6. Real-Time Telemetry & Monitoring Commands

### 1. Check GPU utilization and allocated VRAM
```bash
ssh hpc "ssh GPU-3090-1 nvidia-smi"
```

### 2. Check active PBS batch jobs
```bash
ssh hpc "qstat -a"
```
Or for detailed job reservation:
```bash
ssh hpc "checkjob <JOB_ID>"
```

### 3. Tail live output from running PBS spool
```bash
ssh hpc "ssh GPU-3090-1 'tail -f /var/spool/torque/spool/<JOB_ID>*'"
```

### 4. Stop / Cancel a running job
```bash
ssh hpc "qdel <JOB_ID>"
```
And immediately verify GPU memory is freed:
```bash
ssh hpc "ssh GPU-3090-1 'pkill -9 -f python; nvidia-smi'"
```

---

## 7. Results Consolidation & Organization

After running benchmarks or training on the GPU, run the consolidation engine to structure all artifacts into standardized directories:

```bash
# Run on cluster
ssh hpc "/home/tahamajlesi.ut/.conda/envs/neuralops/bin/python ~/NeuralOps/scripts/benchmarks/consolidate_and_organize_benchmarks.py"

# Sync cleanly to local Mac (excluding heavy model checkpoints)
rsync -avz --exclude 'checkpoints' hpc:~/NeuralOps/results/ results/
```

### Output Layout
- `results/csv/`: Clean, typed CSVs (`attention_zoo_benchmark.csv`, `compressor_zoo_benchmark.csv`, `serving_matrix_benchmark.csv`, `triton_26_kernels_benchmark.csv`).
- `results/tables/`: Ready-to-publish LaTeX tables with English math mode `$ ... $` (`attention_zoo_summary.tex`, etc.).
- `results/reports/`: Executive Markdown dashboard (`BENCHMARK_DASHBOARD.md`).
- `results/checkpoints/`: Model weights (`stress_checkpoint_step_*.pt`).
- `results/master_ledger.sqlite`: Crash-resilient SQLite database with full history.

---

## 8. Diagnostic & Error Troubleshooting Matrix

| Error Symptom | Root Cause | Immediate Fix |
| :--- | :--- | :--- |
| `qsub: Unauthorized Request MSG=group ACL is not satisfied` | User not in `HPC` group for queue `N3090`. | Verify `id` shows `GID 2423 (HPC)`. If not, request admin `usermod -a -G HPC <user>`. |
| `checkjob: Rejection Reasons: [Features: 21]` | Script requested `:gpus=1`. | Change resource request to `#PBS -l nodes=1:ppn=1` without `:gpus=1`. |
| `CUDA initialization: driver on your system is too old (11030)` | Default conda environment has PyTorch 2.4+ (requires CUDA 12). | Use PyTorch 1.12.1 by setting `PYTHONPATH="/home/tahamajlesi.ut/venv_cu113/..."`. |
| `OSError: libmkl_intel_lp64.so: cannot open shared object file` | MKL dynamic libraries missing from `LD_LIBRARY_PATH`. | Add `/share/apps/Anaconda/anaconda3.8/lib` to `LD_LIBRARY_PATH`. |
| `ModuleNotFoundError: No module named 'pkg_resources'` | Setuptools 70+ removed pkg_resources in PyTorch 1.12 `cpp_extension.py`. | Created shim `/home/tahamajlesi.ut/venv_cu113/.../pkg_resources.py` (`import packaging`). |
| `UnicodeEncodeError: 'charmap' codec can't encode character` | Node locale is `iso8859_15` instead of `UTF-8`. | Export `PYTHONIOENCODING=utf-8` and use ASCII labels in scripts. |
| `ImportError: libtiff-56c1bbc6.so.6.2.0: cannot open shared object file` | Broken Pillow in `venv_cu113`. | Removed broken `PIL` in venv so working conda Pillow is used. |
