# 🐳 NeuralOps Containerized GPU Environment

This directory provides containerized, isolated, and bit-exact reproducible execution environments for the **NeuralOps v1.4.0** ecosystem.

Using Docker with the NVIDIA Container Toolkit guarantees:
- Identical CUDA 12.8 toolchain and driver bindings.
- PyTorch 2.11+ with Inductor `torch.compile` pre-configured.
- OpenAI Triton 3.6+ JIT compilation cache persistence.
- Zero local host Python package contamination.

---

## 📋 Prerequisites

1. **Host OS**: Linux (Ubuntu 22.04/24.04 recommended), WSL2 on Windows 11, or compatible POSIX host.
2. **NVIDIA Driver**: Version $\ge 550.00$ (verified on 595.99.02).
3. **Docker Engine**: Version $\ge 24.0$ with Docker Compose v2.
4. **NVIDIA Container Toolkit**: Configured as default Docker runtime (`nvidia-ctk runtime configure --runtime=docker`).

Verify GPU container runtime:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## 🚀 Quick Start

### 1. Build and Start Container

From the repository root or `docker/` folder:

```bash
# Build the GPU image and start interactive container in the background
docker compose -f docker/docker-compose.yml up -d --build
```

### 2. Enter Interactive Shell

```bash
docker compose -f docker/docker-compose.yml exec neuralops-gpu bash
```

Inside the container:
```bash
# Verify environment and CUDA device
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Run canonical invariant count check (131 attention, 72 compression, 26 Triton kernels)
python3 scripts/utils/verify_readme_counts.py
```

---

## 🔬 Reproducing Physical GPU Benchmarks

Inside the running container, execute the canonical benchmark suites:

```bash
# 1. Benchmark all 26 Triton JIT kernels
python3 scripts/benchmarks/benchmark_all_26_kernels.py

# 2. Benchmark all 131 Attention Mechanisms
python3 scripts/benchmarks/benchmark_all_131_attentions.py

# 3. Benchmark all 72 Compression Engines
python3 scripts/benchmarks/benchmark_all_72_compressors.py

# 4. Strict Inductor torch.compile sweep across 131 attention mechanisms
python3 benchmarks/benchmark_torch_compile_131_attentions.py --strict-compile-check

# 5. Isolated 108 Native CUDA C++ kernels classification
python3 scripts/benchmarks/benchmark_all_108_cuda_isolated.py --classify-strict
```

All benchmark artifacts, CSV logs, and JSON summaries are written to `/workspace/results/` and persisted directly onto your host directory via volume mounting.

---

## 🌐 Services and Ports

| Service | Port | Launch Command inside Container | Purpose |
| :--- | :---: | :--- | :--- |
| **Telemetry Web Dashboard** | `8080` | `neuralops web --port 8080` | Real-time NVML GPU monitoring & Roofline charts |
| **JupyterLab Suite** | `8888` | `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root` | Interactive 11 Master Notebooks |

Access the dashboard on your host browser at `http://localhost:8080` and JupyterLab at `http://localhost:8888`.

---

## 🛑 Teardown

```bash
docker compose -f docker/docker-compose.yml down
```
