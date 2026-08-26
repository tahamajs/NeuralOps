# NeuralOps: NeurIPS/ICLR 2026 Reproducible Benchmark Container
FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace:/workspace/Codes

WORKDIR /workspace

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install pinned dependencies
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy codebase
COPY . /workspace/

# Default entrypoint runs a quick smoke test
CMD ["python", "Codes/framework/run_quality_sweep.py", "--max_steps", "1", "--seeds", "42", "--modules", "full", "--models", "bert", "--datasets", "sst2", "--output_dir", "/tmp/smoke_test"]
