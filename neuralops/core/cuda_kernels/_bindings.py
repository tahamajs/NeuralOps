"""
CUDA C++ JIT build loader with persistent compilation caching on /mnt/c.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

if os.path.exists("/usr/local/cuda"):
    if not os.environ.get("CUDA_HOME") or os.environ.get("CUDA_HOME") == "/usr/local":
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
    cuda_bin = "/usr/local/cuda/bin"
    curr_path = os.environ.get("PATH", "")
    if cuda_bin not in curr_path:
        os.environ["PATH"] = f"{cuda_bin}:{curr_path}"

import torch
import torch.utils.cpp_extension as _cpp_ext
from torch.utils.cpp_extension import load_inline

if os.path.exists("/usr/local/cuda"):
    _cpp_ext.CUDA_HOME = "/usr/local/cuda"

logger = logging.getLogger(__name__)

# Persistent cache directory on /mnt/c if available, else ~/.cache
_DEFAULT_CACHE_DIR = (
    "/mnt/c/linux_cache/torch_extensions"
    if os.path.exists("/mnt/c")
    else os.path.expanduser("~/.cache/torch_extensions")
)
_CACHE_DIR = os.environ.get("NEURALOPS_CUDA_CACHE", _DEFAULT_CACHE_DIR)
os.environ.setdefault("TORCH_EXTENSIONS_DIR", _CACHE_DIR)

try:
    Path(_CACHE_DIR).mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.debug(f"Could not create cache dir {_CACHE_DIR}: {e}")

_LOADED_MODULES: dict[str, Any] = {}


def get_cuda_arch_flags() -> list[str]:
    """Auto-detect GPU compute capability and return corresponding nvcc arch flag."""
    if not torch.cuda.is_available():
        return ["-arch=sm_70"]
    try:
        major, minor = torch.cuda.get_device_capability()
        sm = f"sm_{major}{minor}"
        supported = {"sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100"}
        if sm not in supported:
            logger.warning(f"Architecture {sm} not explicitly in supported set, using sm_80 fallback")
            sm = "sm_80"
        return [f"-arch={sm}"]
    except Exception as e:
        logger.warning(f"Failed to detect device capability: {e}. Defaulting to sm_80.")
        return ["-arch=sm_80"]


def load_cuda_module(
    name: str,
    cpp_sources: str,
    cuda_sources: str,
    functions: list[str],
    extra_cuda_cflags: list[str] | None = None,
    force_rebuild: bool = False,
) -> Any | None:
    """
    Load a CUDA C++ module via load_inline with caching and fallback.
    """
    if not force_rebuild and name in _LOADED_MODULES:
        return _LOADED_MODULES[name]

    if not torch.cuda.is_available():
        logger.debug(f"CUDA unavailable; skipping CUDA compilation for '{name}'")
        return None

    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 7:
            logger.info(f"Compute capability {major}.x < 7.0 (pre-Volta); skipping CUDA compilation for '{name}'")
            return None
    except Exception:
        return None

    if shutil.which("nvcc") is None:
        logger.warning(f"nvcc executable not found in PATH; skipping CUDA compilation for '{name}'")
        return None

    arch_flags = get_cuda_arch_flags()
    base_cflags = ["-O3", "--use_fast_math", "-lineinfo"]
    if os.path.exists("/usr/local/cuda/include"):
        base_cflags.append("-I/usr/local/cuda/include")

    all_cuda_cflags = base_cflags + arch_flags + (extra_cuda_cflags or [])
    cpp_cflags = ["-O3"]
    if os.path.exists("/usr/local/cuda/include"):
        cpp_cflags.append("-I/usr/local/cuda/include")

    try:
        mod_cache_dir = Path(_CACHE_DIR) / name
        mod_cache_dir.mkdir(parents=True, exist_ok=True)
        module = load_inline(
            name=name,
            cpp_sources=cpp_sources,
            cuda_sources=cuda_sources,
            functions=functions,
            extra_cuda_cflags=all_cuda_cflags,
            extra_cflags=cpp_cflags,
            verbose=False,
            with_cuda=True,
            build_directory=str(mod_cache_dir),
        )
        _LOADED_MODULES[name] = module
        logger.info(f"✅ CUDA C++ module '{name}' loaded successfully")
        return module
    except Exception as e:
        logger.warning(f"❌ CUDA C++ compilation failed for '{name}': {e}")
        return None
