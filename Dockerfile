# NeuralOps: NeurIPS/ICLR 2026 Reproducible Benchmark Container
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace:/workspace/Codes

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.12 -m pip install --upgrade pip setuptools wheel

# Install PyTorch and Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY . /workspace/

# Default entrypoint runs verification suite
CMD ["bash", "run_full_benchmark.sh"]
