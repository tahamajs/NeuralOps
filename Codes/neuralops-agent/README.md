# NeuralOps GPU Provider Agent & Desktop GUI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Electron: 28+](https://img.shields.io/badge/Electron-28%2B-47848F.svg)](https://www.electronjs.org/)
[![React: 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)

**NeuralOps GPU Provider Agent** is a lightweight daemon and desktop application that enables hardware owners to monetize idle NVIDIA GPU compute on the decentralized NeuralOps cloud compute network.

---

## 🌟 Key Features

- **Automated GPU Provisioning**: Auto-detects NVIDIA GPUs, VRAM capacity, compute capability, and executes MLPerf-style inference benchmarks.
- **Hardware-Isolated Execution**: Securely executes user workloads inside Docker containers equipped with NVIDIA Container Toolkit runtime pass-through, CPU/memory quotas, and read-only root filesystems.
- **Real-Time Telemetry & Sensors**: Streams 1 Hz GPU core load, VRAM allocations, thermal sensors, fan speeds, and power draw metrics.
- **Desktop Electron & React GUI**: Modern desktop dashboard with dark-mode interface, frameless windowing, system tray integration, and real-time earnings notifications.
- **HTTP/SOCKS5 Networking Bridge**: Embedded HTTP-to-SOCKS5 proxy (`bridge.py`) for SSL tunneling, container networking, and seamless cross-origin communication.
- **Decentralized Escrow & Instant Payouts**: Automated escrow verification, dispute resolution hooks, and real-time balance tracking.
- **Cross-Platform Installers**: Native packaging for macOS (`.dmg`), Windows (`.exe` NSIS installer), and Linux (`.AppImage` & `.deb`).

---

## 🏗️ Architecture

```
Codes/neuralops-agent/
├── agent/                         # Core Python Daemon & Worker Logic
│   ├── main.py                    # Entry point orchestrator
│   ├── config.py                  # YAML and CLI configuration parser
│   ├── gpu_detector.py            # Hardware detection & MLPerf benchmarks
│   ├── gpu_monitor.py             # 1 Hz live hardware sensor collector
│   ├── docker_manager.py          # Container sandbox lifecycle & limits
│   ├── job_executor.py            # Job execution & log streaming
│   ├── websocket_client.py        # Persistent WebSocket connection
│   ├── api_client.py              # REST API client for backend
│   └── network_tester.py          # Bandwidth and latency speed tests
│
├── neuralops-gui/                 # Electron + React Desktop App
│   ├── bridge.py                  # HTTP-to-SOCKS5 proxy tunnel bridge
│   ├── electron/
│   │   ├── main.js                # Electron main process & IPC handlers
│   │   └── preload.js             # Secure contextBridge API bindings
│   ├── src/                       # React 18 frontend components
│   │   ├── components/            # Dashboard, GPUDetails, Jobs, Earnings, etc.
│   │   ├── services/              # api.js REST & websocket.js clients
│   │   ├── context/               # Theme and Notification context providers
│   │   └── styles/                # Tailwind & global CSS
│   ├── webpack.config.js          # Webpack 5 bundle configuration
│   └── package.json               # Node dependencies & electron-builder
│
├── scripts/                       # Packaging & Installation Utilities
│   ├── build.py                   # Multi-platform binary and installer builder
│   ├── install.sh                 # Linux/macOS one-line installer
│   └── install.ps1                # Windows PowerShell installer
│
├── tests/                         # Test Suite
│   └── test_gui_bridge.py         # Comprehensive unit tests for GUI bridge & proxies
│
├── agent-config.yaml              # Default agent configuration template
├── pyinstaller.spec               # PyInstaller standalone executable spec
├── setup.py                       # Python setuptools packaging
└── requirements.txt               # Python package dependencies
```

---

## 📋 Prerequisites

1. **Operating System**: Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+), macOS (macOS 12+), or Windows 10/11 (64-bit).
2. **GPU Hardware**: NVIDIA GPU with CUDA Compute Capability >= 6.0 (e.g. GTX 10-series, RTX 20/30/40 series, A100/H100/L40S).
3. **NVIDIA Drivers**: Version 525.xx or higher with CUDA 12.x support.
4. **Docker Engine & NVIDIA Container Toolkit**:
   - `docker` (v20.10+)
   - `nvidia-container-toolkit` (`nvidia-docker2`)
5. **Python**: Python 3.10 or higher.
6. **Node.js** (for GUI development/builds): Node.js 18+ and npm 9+.

---

## 🚀 Quick Start

### 1. Installation via Python Pip

```bash
# Clone the repository
git clone https://github.com/neuralops/neuralops-agent.git
cd neuralops-agent

# Install dependencies and package
pip install -e .
```

### 2. Configure Your Provider Credentials

Copy the template configuration:
```bash
cp agent-config.yaml ~/.neuralops-agent/agent-config.yaml
```

Set your API key:
```bash
export NEURALOPS_API_KEY="your_api_key_here"
```

### 3. Start the Daemon

```bash
neuralops-agent --config agent-config.yaml
```

---

## 💻 Desktop GUI Application

The NeuralOps desktop application provides a provider dashboard with live telemetry and system tray controls.

### Running the GUI in Development Mode

```bash
cd neuralops-gui
npm install
npm run dev
```

### Building the Production Desktop App

```bash
# Build React bundle
npm run build

# Package Desktop App for current OS
npm run dist

# Or build platform-specific packages:
npm run dist:mac     # macOS .dmg
npm run dist:win     # Windows .exe installer
npm run dist:linux   # Linux .AppImage / .deb
```

---

## 🔧 Networking & GUI Bridge (`bridge.py`)

The embedded HTTP to SOCKS5 bridge provides proxy tunneling:

```bash
python neuralops-gui/bridge.py --http-port 8080 --socks-port 1080
```

### Programmatic Python Usage

```python
from bridge import HTTPToSOCKS5Bridge, BridgeConfig

config = BridgeConfig(
    http_host="127.0.0.1",
    http_port=8080,
    socks_host="127.0.0.1",
    socks_port=1080,
)
bridge = HTTPToSOCKS5Bridge(config)
bridge.start()  # Runs in background thread

# ... later ...
bridge.stop()
```

---

## 📦 Packaging & Release Pipeline (`scripts/build.py`)

The unified build script builds both standalone PyInstaller agent binaries and Electron installers:

```bash
# Build for current host platform
python scripts/build.py

# Explicit platform builds
python scripts/build.py --platform macos
python scripts/build.py --platform windows
python scripts/build.py --platform linux
```

---

## 🧪 Testing

Run the test suite using `pytest` or Python's built-in `unittest`:

```bash
# Run tests with pytest
python -m pytest tests/ -v

# Run tests with unittest
python -m unittest discover tests -v
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
