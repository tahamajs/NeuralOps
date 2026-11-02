---
name: hpc-sharif-cluster
description: "Run GPU-accelerated tests, CUDA compilation, Triton JIT execution, and DoD empirical measurements on the Sharif University HPC cluster (Rocks 6.1, Torque PBS)."
version: "1.0.0"
tags: ["neuralops", "hpc", "sharif", "cluster", "gpu", "verification"]
---

# Skill: Sharif University HPC Cluster

**Purpose:** Run GPU-accelerated tests, CUDA compilation, Triton JIT execution, and DoD empirical measurements on the Sharif University High Performance Computing Center cluster.

**Related sections:** AGENTS.md §3.1 (remote GPU execution), §3.4 (Sharif cluster), §4 (verification contract).

**Activation:** Load this skill before submitting any GPU job to `hpc.itc.sharif.edu`. Do not use for CPU-only work — use local environment per §3.1.

---

## 1. Cluster Overview

| Property | Value |
|---|---|
| Hostname | `hpc.itc.sharif.edu` |
| SSH alias | `hpc` (configured in `~/.ssh/config`) |
| Login node | `login-3` |
| OS | Rocks 6.1 (CentOS 6 derivative) |
| Scheduler | Torque PBS (`qsub`, `qstat`, `qdel` at `/opt/torque/bin/`) |
| User | `tahamajlesi.ut` |
| Home | `/home/tahamajlesi.ut` |
| Quota | 60 GB free (soft), 24 TB shared filesystem |
| Max cores per job | 80 |
| Max cores total | 400 |
| Max RAM per job | 24 GB recommended (varies by queue) |

**Note:** `python3` is NOT in default PATH. All Python work uses conda.

## 2. Queue Reference

Run `qstat -Q` **before** submitting to know which queues are enabled right now. Queues change state daily.

> **CRITICAL**: For complete operational instructions, zero-drop SSH configuration, and environment setup, read `HPC_GPU_MASTER_GUIDE.md` in the repository root.

| Queue | GPU | VRAM | Status / Typical use |
|---|---|---|---|
| **`N3090`** | **NVIDIA RTX 3090** | **24 GB** | **ACTIVE PRIMARY**. Compute capability 8.6, 350W TDP. Request: `#PBS -l nodes=1:ppn=1` (do NOT use `:gpus=1`) |
| `SP` | RTX 2070 Super | 8 GB | Active secondary for lightweight models |
| `cuda7` | Tesla C2075 | 5 GB | Disabled / Deprecated |
| `cuda9` | GTX 1080 Ti | 11 GB | Disabled by HPC admin |
| `DP` | Titan RTX | 24 GB | Disabled by HPC admin |
| `DDP` | Quadro RTX 8000 | 48 GB | Disabled by HPC admin |

`qstat -Q` output columns: `Max | Tot | Ena | Str | Que | Run | Hld | Wat | Trn | Ext | T`
- `Ena=yes` and `T=@` (or `E`) means the queue accepts jobs.
- If `Ena=no`, the queue is disabled — pick a different one.

## 3. Environment Setup (One-Time)

### 3.1 — Verify SSH

```bash
ssh hpc "whoami; pwd; hostname; df -h ~ | tail -1"
```
Expected: `tahamajlesi.ut`, `/home/tahamajlesi.ut`, `login-3`.

### 3.2 — Upload code
From the local Mac (not from SSH):

```bash
ssh hpc "mkdir -p ~/NeuralOps"
rsync -avz --progress \
  --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='*.log' --exclude='kaggle_*.txt' \
  --exclude='results/' --exclude='audit/' --exclude='academic/' \
  --exclude='paper/' --exclude='thesis_final/' --exclude='notebooks/' \
  --exclude='docs/' --exclude='build/' --exclude='dist/' \
  --exclude='*.egg-info' \
  /Users/tahamajs/dev/Project/ hpc:~/NeuralOps/
```
Verify: `ssh hpc "cd ~/NeuralOps && ls; du -sh ."`

### 3.3 — Create conda environment
```bash
ssh hpc "/share/apps/Anaconda/anaconda3.8/bin/conda create -n neuralops python=3.10 -y"
```
Conda paths available in `/share/apps/Anaconda/`: `anaconda3.8` (recommended), `anaconda3.11`, `anaconda2024`.

### 3.4 — Install dependencies
```bash
ssh hpc "source /share/apps/Anaconda/anaconda3.8/bin/activate neuralops && \
  cd ~/NeuralOps && \
  pip install --upgrade pip && \
  pip install torch triton pytest transformers datasets huggingface_hub && \
  pip install -e . --no-deps"
```
Verify: `ssh hpc "source /share/apps/Anaconda/anaconda3.8/bin/activate neuralops && python -c 'import neuralops; print(neuralops.__version__)'"`

Expected: `2.0.0`.

### 3.5 — One-shot bootstrap script
See `scripts/setup_env.sh` in this skill for a replayable version.

## 4. Running GPU Tests

### 4.1 — Verify credit balance
```bash
ssh hpc "grep 'credit Quota' /etc/motd 2>/dev/null || echo 'check via https://ssh.hpc.sharif.ir → Wallet'"
```
If credit Quota = 0.0 → STOP. Ask user to charge wallet (~300,000 tomans for ~20 GPU-hours recommended).

### 4.2 — Write run.sh
Copy `scripts/run.sh` from this skill, adjust the `#PBS -q` line to match a queue that is currently `Ena=yes` in `qstat -Q` output.

Common queue substitutions:
- `#PBS -q cuda7` — Tesla K80 (12 GB), always available
- `#PBS -q DP` — Titan RTX (24 GB), best for NeuralOps when enabled
- `#PBS -q DDP` — Quadro RTX 8000 (48 GB), largest model

### 4.3 — Submit
```bash
ssh hpc "cd ~/NeuralOps && qsub run.sh"
```
Output: `<JOB_ID>.login-3` (e.g., `12345.login-3`). Record `JOB_ID`.

### 4.4 — Monitor
```bash
ssh hpc "qstat -u tahamajlesi.ut"      # all your jobs
ssh hpc "qstat -f <JOB_ID>"            # one job
```
Status: `Q`=queued, `R`=running, `E`=exiting, `C`=completed.
Poll every 60s max (Torque rate-limits faster polling).

### 4.5 — Read results
```bash
ssh hpc "cd ~/NeuralOps && cat neuralops_gpu_verify.o<JOB_ID>"
ssh hpc "cd ~/NeuralOps && cat neuralops_gpu_verify.e<JOB_ID>"
```

### 4.6 — Fetch locally
```bash
rsync -avz hpc:~/NeuralOps/neuralops_gpu_verify.* /Users/tahamajs/dev/Project/hpc_results/
```
Add `hpc_results/` to `.gitignore` before committing (see §7).

### 4.7 — Cancel (if needed)
```bash
ssh hpc "qdel <JOB_ID>"
```

## 5. Verification Contract
Per AGENTS.md §4, any GPU job on Sharif HPC must produce:
1. Raw `qstat -f <JOB_ID>` before and after completion
2. Full contents of `.o<JOB_ID>` (stdout) and `.e<JOB_ID>` (stderr) pasted in the report
3. DoD metrics table with MEASURED / DEFERRED per row (never fabricate)
4. CPU-only tests must have been run first locally (per §3.1 — do not waste GPU credit on tests that work on CPU)
5. Do not claim "GPU verified" without pasted pytest summary line.

## 6. Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `qsub: Unknown queue name` | Queue renamed/removed | `qstat -Q` again, pick available |
| `qsub: Job violates queue/server policy` | Over-request (cores/GPU/mem) | Reduce ppn, gpus, or mem in run.sh |
| `qsub: Unauthorized Request` | Credit = 0 | Charge wallet via panel |
| Job stuck in Q > 30 min | Queue full | Wait for off-hours, try test-gpu |
| `torch.cuda.is_available() = False` in job | GPU not allocated | Ensure `#PBS -l gpus=1` in run.sh |
| `conda: command not found` | Path wrong | Use full path `/share/apps/Anaconda/anaconda3.8/bin/conda` |
| `ModuleNotFoundError: neuralops` | Not installed in env | `pip install -e . --no-deps` in activated env |
| Permission denied on SSH | Password wrong | Reset via https://ssh.hpc.sharif.ir → Forgot password |
| Connection times out after idle | TCP keepalive | Add `ServerAliveInterval 60` to `~/.ssh/config` |

## 7. Repository Hygiene
- Add `hpc_results/` to `.gitignore` (never commit result files)
- Never commit `~/NeuralOps/run.sh` if it contains user-specific paths (it's a local override; the canonical template lives in this skill)
- Do NOT commit `.o<JOB_ID>` or `.e<JOB_ID>` files — they are transient artifacts

## 8. Primary Remote Execution Architecture
Sharif HPC is the primary and sole approved remote GPU host for NeuralOps:
- DoD verification on large models (Titan RTX 24GB, Quadro 48GB)
- CUDA C++ compilation and Triton JIT parity verification
- Long-running distributed tests and kernel autotuning

## 9. Cost Awareness
GPU-hours are billed against the wallet balance. Before submitting a job:
1. Estimate runtime (typical: 5-30 min for Triton parity, 10-60 min for E2E DoD)
2. Verify credit > (estimated minutes × hourly_rate)
3. Do NOT submit a job "just to see if it works" — use test-gpu for that (lower rate)
4. Cancel promptly if you see the job is not producing useful output

## 10. Mandatory Reading Before First GPU Job
- AGENTS.md §0 (Core Operating Principles)
- AGENTS.md §3.1 (remote GPU execution rule)
- AGENTS.md §3.4 (Sharif cluster)
- AGENTS.md §4.2 (kernel verification contract)
- This skill file (§1–§9)

Failure to read these results in wasting GPU credit on jobs that would have failed locally or orphaned jobs left running.
