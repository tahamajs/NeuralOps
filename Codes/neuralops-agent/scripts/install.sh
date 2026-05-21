#!/bin/bash
set -e
echo "=== NeuralOps GPU Provider Agent Installer ==="

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Python 3.10+ required. Install it first."
    exit 1
fi

# Check NVIDIA drivers
if ! command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA drivers not found. Install CUDA drivers first."
    exit 1
fi

# Check Docker
if ! command -v docker &>/dev/null; then
    echo "Docker not found. Installing..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
fi

# Install NVIDIA Container Toolkit
if ! docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker.gpg
    echo "deb [signed-by=/usr/share/keyrings/nvidia-docker.gpg] https://nvidia.github.io/nvidia-docker/$distribution/$(arch) /" | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt update && sudo apt install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# Install agent
INSTALL_DIR="$HOME/.neuralops-agent"
mkdir -p "$INSTALL_DIR"
python3 -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"
pip install neuralops-agent

echo ""
echo "✅ Installation complete!"
echo "   Run: neuralops-agent --api-key YOUR_KEY"