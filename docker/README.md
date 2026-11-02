# 🐳 NeuralOps Containerized Execution & Verification Environment

This directory provides containerized, isolated, and bit-exact reproducible execution environments for **NeuralOps v2.0.2**.

Using Docker with the NVIDIA Container Toolkit guarantees:
- Identical CUDA 12.8 toolchain and driver bindings.
- PyTorch with Inductor `torch.compile` pre-configured.
- OpenAI Triton JIT compilation cache persistence.
- Zero local host Python package contamination.

---

## 📋 Prerequisites

1. **Host OS**: Linux (Ubuntu 22.04/24.04 recommended), WSL2 on Windows 11, or compatible POSIX host.
2. **NVIDIA Driver**: Version $\ge 550.00$ (verified on 595.99.02 for Ada Lovelace / Hopper / Blackwell).
3. **Docker Engine**: Version $\ge 24.0$ with Docker Compose v2.
4. **NVIDIA Container Toolkit**: Configured as default Docker runtime (`nvidia-ctk runtime configure --runtime=docker`).

Verify GPU container runtime:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 One-Click Autonomous GPU Execution (Recommended)

When connecting to the GPU machine, you can run the entire pipeline end-to-end with a single command. It builds the container, verifies the environment, executes all benchmarks across all 131 attentions and 72 compressors, runs the smart GPU trainer, and exports the final paper tables and figures:

```bash
# From repository root on GPU host:
docker compose -f docker/docker-compose.yml run --rm neuralops-gpu /workspace/docker/run_gpu_pipeline.sh
```

All benchmark artifacts, SQLite ledgers, CSV logs, LaTeX tables, and vector figures are persisted directly into your host directory via volume mounting.

---

## 🛠️ Interactive Usage

### 1. Build and Start Container

From the repository root:
```bash
# Start GPU container in background
docker compose -f docker/docker-compose.yml up -d --build neuralops-gpu
```

*(For systems without NVIDIA GPU, start CPU container: `docker compose -f docker/docker-compose.yml up -d --build neuralops-cpu`)*

### 2. Enter Interactive Shell

```bash
docker compose -f docker/docker-compose.yml exec neuralops-gpu bash
```

Inside the container:
```bash
# Verify environment and CUDA device
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## 🔬 Running Individual Subsystems

Inside the container, you can run individual suites:

```bash
# 1. Master benchmark suite (131 Attentions + 72 Compressors + Kernel 3-level fallback)
python3 scripts/benchmarks/gpu_master_suite.py --category all --device cuda --export

# 2. Fast dry-run smoke test (verifies all pipelines in seconds)
python3 scripts/benchmarks/gpu_master_suite.py --dry-run --export

# 3. Smart GPU-Maximizing Trainer (VRAM saturation, async prefetching, OOM auto-recovery)
python3 scripts/benchmarks/smart_gpu_trainer.py

# 4. Export updated LaTeX tables for paper
python3 scripts/export_paper_tables.py

# 5. Generate ArXiv vector figures (PDF + PNG)
python3 scripts/generate_arxiv_figures.py
```

---

## 🌐 Services and Ports

| Service | Port | Launch Command inside Container | Purpose |
| :--- | :---: | :--- | :--- |
| **Telemetry Web Dashboard** | `8080` | `neuralops web --port 8080` | Real-time NVML GPU monitoring & Roofline charts |
| **JupyterLab Suite** | `8888` | `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root` | Interactive 29 Master Notebooks |

Access the dashboard on your host browser at `http://localhost:8080` and JupyterLab at `http://localhost:8888`.

---

## 🛑 Teardown

```bash
docker compose -f docker/docker-compose.yml down
```
