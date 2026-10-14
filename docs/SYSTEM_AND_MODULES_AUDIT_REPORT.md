# Deep Diagnostic System & Python Modules Audit Report

**Date of Execution:** 2026-10-14  
**Investigated Host:** `tahoora` (Ubuntu 26.04.1 LTS, Native Linux x86_64, Dual-Host Cross-Comparison with `gpu` host)  
**Investigation Mode:** Strict Read-Only Diagnostic Audit  
**Target Environment:** `/home/tahoora/Project/.venv` (Python 3.12.15), System Python 3.14.4, cross-referenced with `gpu` (`/mnt/d/venv` Python 3.14.7)  

---

## 1. Executive Summary

A comprehensive, read-only diagnostic audit of the compute infrastructure, CUDA runtime and compilation stack, dynamic link libraries (`.so`), virtual environments, and PyTorch acceleration backends was conducted on `tahoora` (with cross-reference to the Windows dual-boot NTFS volume on `gpu`).

### Root-Cause Analysis & Key Findings:
1. **Hardware Compute Capability Mismatch for `flash-attn`**:
   - `tahoora` is equipped with an **NVIDIA GeForce GTX 1060 6GB** with **Pascal Architecture** (Compute Capability **6.1**).
   - `flash-attn` (Tri Dao / FlashAttention-2) strictly enforces hardware support for **Ampere, Ada Lovelace, or Hopper architectures (Compute Capability $\ge 8.0$)**. It is mathematically and architecturally incompatible with Pascal GPUs (CC 6.1). Any attempt to install or execute `flash-attn` kernels on GTX 1060 is fundamentally unsupported by the CUDA PTX/SASS kernel instructions.
2. **Missing Standalone CUDA Toolkit (`nvcc` & Compilation Headers)**:
   - Neither `tahoora` nor the secondary host possesses a local CUDA Toolkit in `/usr/local/cuda`. `nvcc` is missing from `PATH` and disk.
   - When `pip` attempts to build source distributions (such as `flash-attn` or `xformers`), setup fails immediately with `FileNotFoundError: '/usr/local/cuda-12.8/bin/nvcc'` or `CUDA_HOME is not set`.
3. **Dynamic Library (`.so`) Preloading & Search Path Separation**:
   - Runtime libraries (`libcudart.so.12`, `libcublasLt.so.12`, `libcudnn.so.9`) exist inside Python wheel directories (`site-packages/nvidia/*/lib`), but neither `LD_LIBRARY_PATH` nor `/etc/ld.so.conf.d/` links them to the host loader.
   - PyTorch bundles its own isolated runtime links, but foreign modules relying on `ctypes.CDLL` or system-level dynamic linker calls fail to locate `libcudart.so.12` and `libcublasLt.so` unless `LD_LIBRARY_PATH` points directly to the bundled wheel libraries.
4. **Python Version Ecosystem & Wheel Availability (Python 3.12 vs 3.14)**:
   - On `tahoora`, the virtual environment is configured with **Python 3.12.15**, which has wide pre-built wheel availability.
   - However, on systems attempting to run bleeding-edge **Python 3.14 (`cp314`)**, PyPI binary wheels do not yet exist for heavy C++/CUDA extensions (`flash-attn`, `xformers`, `decord`, `av`), causing pip to fall back to source tarballs (`.tar.gz`), which crash due to missing `nvcc` and C++20 ABI mismatches with GCC 15.
5. **Project Native Attention Backend Independence**:
   - Inspection of `~/Project/neuralops` confirms that the codebase **does not require the external `flash-attn` package**.
   - Instead, the framework cleanly utilizes PyTorch's native C++ scaled dot-product attention (`torch.nn.functional.scaled_dot_product_attention`, SDPA), which dynamically dispatches to memory-efficient math and FP16 kernels on the GTX 1060 without external dependencies.

---

## 2. Hardware, GPU & Operating System Profile

| Parameter | Specification on `tahoora` | Comparative Note (`gpu` server) |
| :--- | :--- | :--- |
| **GPU Name** | NVIDIA GeForce GTX 1060 | NVIDIA GeForce RTX 4070 Laptop GPU |
| **Microarchitecture** | **Pascal** | **Ada Lovelace** |
| **Compute Capability** | **(6, 1)** (SM 6.1) | **(8, 9)** (SM 8.9) |
| **Total VRAM** | **6144 MiB (6.0 GB GDDR5)** | 8188 MiB (8.0 GB GDDR6) |
| **VRAM In-Use** | ~3557 MiB (active RoBERTa sweep PID 776349) | ~10 MiB (idle) |
| **NVIDIA Driver Version** | **580.178.04** | 595.99.02 |
| **Max Supported CUDA (Driver)** | **CUDA 13.0** | CUDA 13.2 |
| **Host OS** | Ubuntu 26.04.1 LTS (Resolute Raccoon) | Ubuntu 26.04.1 LTS |
| **Linux Kernel** | `7.0.0-38-generic #38-Ubuntu x86_64` | `7.0.0-38-generic #38-Ubuntu x86_64` |
| **System Environment** | **Native Linux** (Bare metal, ext4 root) | Native Linux dual-boot |
| **Host CPU** | Intel x86_64 (Multi-core) | 13th Gen Intel Core i7-13620H (16 vCPUs) |
| **System RAM** | 14 GiB Total (11 GiB available) | 14 GiB Total (11 GiB available) |
| **Swap Space** | 4.0 GiB (1.2 GiB in use) | 4.0 GiB |

### Analysis of Storage & Mount Topology:
- Root partition on `tahoora` is native **ext4** (`/dev/sda4`, 77GB, 84% utilized).
- A dangling symlink `/home/tahoora/Project/SVD_Project -> /mnt/d/SVD_Project` previously pointed to an unmounted NTFS volume (`/mnt/d`), which exists on the dual-boot drive.
- Running Python environments or compilations across NTFS/drvfs mounts introduces high latency, locks on file creation, and breaks POSIX symlink semantics. Storing the repository and virtual environment on native ext4 (`~/Project`) is the correct configuration.

---

## 3. CUDA Layer, Compilers & Shared Libraries (`.so`) Forensics

### 3.1 Environment Variables
```bash
CUDA_HOME:       <UNSET>
CUDA_PATH:       <UNSET>
LD_LIBRARY_PATH: <UNSET>
PATH:            /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
```

### 3.2 Compilers & Build Toolchain
- **`gcc` / `g++`**: `/usr/bin/gcc`, `/usr/bin/g++` (Ubuntu 15.2.0-16ubuntu1)
- **`cmake`**: Version `4.2.3`
- **`ninja`**: Available on system (`/usr/bin/ninja`, version 1.13.2)
- **`nvcc`**: **MISSING**. No CUDA Toolkit installed in `/usr/local/cuda` or `/usr/local/cuda-12.*`.

### 3.3 Dynamic Shared Library (`.so`) Locations
Querying `/etc/ld.so.cache` via `ldconfig -p` reveals **no system-wide registration** of NVIDIA CUDA runtime libraries. However, pip-installed wheels provide them inside the virtual environment:

- **`libcudart.so`**:
  - `/home/tahoora/Project/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12`
  - `/home/tahoora/Project/.venv/lib/python3.12/site-packages/torchvision.libs/libcudart.7ec1eba6.so.12`
- **`libcublasLt.so`**:
  - `/home/tahoora/Project/.venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12`
- **`libcudnn.so`**:
  - `/home/tahoora/Project/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9`
  - `/home/tahoora/Project/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_cnn.so.9`

### Why `ctypes.CDLL` and `_preload_cuda_deps` Fail:
Standard Linux shared library resolution follows `DT_RPATH` $ightarrow$ `LD_LIBRARY_PATH` $ightarrow$ `/etc/ld.so.cache` $ightarrow$ `/usr/lib`. When third-party packages or diagnostic scripts call `ctypes.CDLL("libcudart.so.12")` without explicit directory qualification, the dynamic loader fails because the `nvidia/*/lib` subdirectories are not added to `LD_LIBRARY_PATH`. PyTorch avoids this internally by programmatically calling `dlopen()` on absolute paths during package initialization.

---

## 4. Python Environment & Path Collision Analysis

### 4.1 Virtual Environment Architecture
- **Active Environment**: `/home/tahoora/Project/.venv`
- **Python Version**: **Python 3.12.15** (packaged by Ubuntu GCC 15.2.0)
- **Base Executable**: `/usr/bin/python3.12`
- **Isolated Site-Packages**: `/home/tahoora/Project/.venv/lib/python3.12/site-packages`
- **Isolation Verification**: `include-system-site-packages = false`. No Conda collisions or site-package pollution from `/home/tahoora/.local` exists.

### 4.2 Bleeding-Edge Python 3.14 (`cp314`) Collision Analysis
- On systems running Python 3.14 (such as the base Ubuntu 26.04 system python or Miniconda 3.14), binary wheels on PyPI are overwhelmingly absent for packages that rely on PyTorch C++ extensions.
- When `pip install flash-attn` or `pip install xformers` is invoked under `cp314`:
  1. Pip checks PyPI for `*-cp314-cp314-linux_x86_64.whl` and receives HTTP 404.
  2. Pip downloads `flash_attn-2.8.3.tar.gz`.
  3. Pip executes `pyproject.toml` / `setup.py build_ext`.
  4. Build fails immediately because `setup.py` searches for `nvcc` in `$CUDA_HOME/bin/nvcc` or `/usr/local/cuda/bin/nvcc`, raising `FileNotFoundError`.

---

## 5. Module-by-Module Diagnostic Audit Table

All tests executed via direct import and live tensor verification inside `/home/tahoora/Project/.venv`:

| Module | Status | Version | Installation Path | GPU Compute / Diagnostics |
| :--- | :---: | :---: | :--- | :--- |
| **`torch`** | ✅ Healthy | `2.5.1+cu121` | `.venv/lib/python3.12/site-packages/torch` | CUDA 12.1, cuDNN 90100, **FP16 MatMul GPU Success**, Device Capability `(6, 1)` |
| **`torchvision`** | ✅ Healthy | `0.20.1+cu121` | `.venv/lib/python3.12/site-packages/torchvision` | Bundled with CUDA 12.1 runtime libraries |
| **`torchaudio`** | ✅ Healthy | `2.5.1+cu121` | `.venv/lib/python3.12/site-packages/torchaudio` | Operational |
| **`flash_attn`** | ❌ Incompatible | Not Installed | N/A | **Hardware Block:** GTX 1060 (Pascal SM 6.1) is unsupported (requires $\ge$ SM 8.0) |
| **`xformers`** | ⚠️ Not Installed | Not Installed | N/A | Not required; PyTorch native SDPA provides equivalent functionality |
| **`triton`** | ✅ Healthy | `3.1.0` | `.venv/lib/python3.12/site-packages/triton` | Pre-built Triton compiler for PyTorch 2.5.1 |
| **`bitsandbytes`** | ⚠️ Not Installed | Not Installed | N/A | Optional (used for 8-bit/4-bit quantization) |
| **`diffusers`** | ⚠️ Not Installed | Not Installed | N/A | Not specified in `neuralops` core NLP requirements |
| **`transformers`**| ✅ Healthy | `4.46.3` | `.venv/lib/python3.12/site-packages/transformers` | Operational |
| **`accelerate`** | ✅ Healthy | `1.15.0` | `.venv/lib/python3.12/site-packages/accelerate` | Operational |
| **`safetensors`** | ✅ Healthy | `0.8.0` | `.venv/lib/python3.12/site-packages/safetensors` | Operational |
| **`einops`** | ✅ Healthy | `0.8.2` | `.venv/lib/python3.12/site-packages/einops` | Operational |
| **`omegaconf`** | ✅ Healthy | `2.3.1` | `.venv/lib/python3.12/site-packages/omegaconf` | Operational |
| **`cv2`** (OpenCV)| ✅ Healthy | `5.0.0` | `.venv/lib/python3.12/site-packages/cv2` | Operational |
| **`decord`** | ⚠️ Not Installed | Not Installed | N/A | Video decoding library; optional |
| **`imageio`** | ✅ Healthy | `2.38.1` | `.venv/lib/python3.12/site-packages/imageio` | Operational |
| **`av`** (PyAV) | ⚠️ Not Installed | Not Installed | N/A | Video container library; optional |
| **`scipy`** | ✅ Healthy | `1.18.1` | `.venv/lib/python3.12/site-packages/scipy` | Operational |
| **`numpy`** | ✅ Healthy | `2.5.2` | `.venv/lib/python3.12/site-packages/numpy` | Operational |
| **`PIL`** (Pillow) | ✅ Healthy | `12.3.0` | `.venv/lib/python3.12/site-packages/PIL` | Operational |
| **`pip`** | ✅ Healthy | `26.2.1` | `.venv/lib/python3.12/site-packages/pip` | Operational |
| **`setuptools`** | ✅ Healthy | `78.1.0` | `.venv/lib/python3.12/site-packages/setuptools` | Operational |
| **`wheel`** | ✅ Healthy | `0.48.0` | `.venv/lib/python3.12/site-packages/wheel` | Operational |
| **`packaging`** | ✅ Healthy | `26.3` | `.venv/lib/python3.12/site-packages/packaging` | Operational |
| **`ninja`** | ⚠️ System Only | System binary | `/usr/bin/ninja` | Python wrapper package not in venv; system binary present |
| **`psutil`** | ✅ Healthy | `7.2.2` | `.venv/lib/python3.12/site-packages/psutil` | Operational |

---

## 6. Project Codebase & Attention Backend Inspection

An exhaustive AST and source code grep across `~/Project/neuralops` reveals:
1. **Scaled Dot-Product Attention Usage**: Over 30 modules (`mqa.py`, `qcqa_attention.py`, `gqa_attention.py`, `fusion_attn.py`, `hybrid_attention.py`, `infini_attention.py`) invoke:
   ```python
   out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=...)
   ```
2. **PyTorch SDPA Capabilities on Pascal (GTX 1060)**:
   - `torch.backends.cuda.flash_sdp_enabled()`: `True` (enabled, but falls back to Math/Mem-Efficient on CC < 8.0)
   - `torch.backends.cuda.mem_efficient_sdp_enabled()`: `True` (fully accelerated via cutlass kernels on Pascal SM 6.1)
   - `torch.backends.cuda.math_sdp_enabled()`: `True`
3. **Verdict**: The project **does NOT require external `flash-attn` compilation**. Attempts to force `flash-attn` installation on this machine are both unnecessary and hardware-impossible.

---

## 7. Definitive Step-by-Step Remediation Plan

To ensure maximum runtime stability and eliminate any future warnings or missing auxiliary libraries:

### Step 1: Export NVIDIA Library Paths to Shell
Prevent any foreign library loader failures by configuring `LD_LIBRARY_PATH` in `~/.bashrc`:
```bash
echo 'export LD_LIBRARY_PATH="/home/tahoora/Project/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/tahoora/Project/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/tahoora/Project/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
```

### Step 2: Install Python-Level Build and Auxiliary Video Tools
If full video and vision benchmark suites are desired:
```bash
source ~/Project/.venv/bin/activate
pip install ninja bitsandbytes diffusers av
```

### Step 3: Maintain Dedicated Native Linux (ext4) Workspaces
Avoid mounting or running virtual environments on NTFS (`/mnt/d` or `/mnt/c`). Always run workflows from `/home/tahoora/Project` to avoid filesystem locks and preserve maximum NVMe/SATA I/O bandwidth.
