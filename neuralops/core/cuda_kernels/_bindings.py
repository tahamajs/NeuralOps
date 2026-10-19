# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CUDA C++ JIT Module Loader

Dynamic compilation loader with caching and device capability validation for native CUDA C++ kernels.
"""

from __future__ import annotations

import hashlib
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

_candidate_cache = os.environ.get("NEURALOPS_CUDA_CACHE")
if not _candidate_cache:
    if os.path.exists("/mnt/d/torch_extensions"):
        _candidate_cache = "/mnt/d/torch_extensions"
    elif os.path.exists("/mnt/d/cache/torch_extensions"):
        _candidate_cache = "/mnt/d/cache/torch_extensions"
    else:
        _candidate_cache = os.path.expanduser("~/.cache/torch_extensions")

try:
    Path(_candidate_cache).mkdir(parents=True, exist_ok=True)
    test_file = Path(_candidate_cache) / ".write_test"
    test_file.touch()
    test_file.unlink()
    _CACHE_DIR = _candidate_cache
except Exception:
    _CACHE_DIR = os.path.expanduser("~/.cache/torch_extensions")
    Path(_CACHE_DIR).mkdir(parents=True, exist_ok=True)

os.environ["TORCH_EXTENSIONS_DIR"] = _CACHE_DIR

_LOADED_MODULES: dict[Any, Any] = {}
_FAILED_MODULES: set[str] = set()


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
    extra_cflags: list[str] | None = None,
    force_rebuild: bool = False,
) -> Any | None:
    """
    Load a CUDA C++ module via load_inline with content-hash caching, negative caching, and fallback.
    """
    if not force_rebuild and name in _FAILED_MODULES:
        return None

    if not torch.cuda.is_available():
        logger.debug(f"CUDA unavailable; skipping CUDA compilation for '{name}'")
        _FAILED_MODULES.add(name)
        return None

    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 7:
            logger.info(f"Compute capability {major}.x < 7.0 (pre-Volta); skipping CUDA compilation for '{name}'")
            _FAILED_MODULES.add(name)
            return None
    except Exception:
        _FAILED_MODULES.add(name)
        return None

    if shutil.which("nvcc") is None:
        logger.warning(f"nvcc executable not found in PATH; skipping CUDA compilation for '{name}'")
        _FAILED_MODULES.add(name)
        return None

    root_include = str(Path(__file__).parent.resolve())
    arch_flags = get_cuda_arch_flags()
    base_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        f"-I{root_include}",
    ]
    if os.path.exists("/usr/local/cuda/include"):
        base_cuda_cflags.append("-I/usr/local/cuda/include")

    all_cuda_cflags = base_cuda_cflags + arch_flags + (extra_cuda_cflags or [])
    cpp_cflags = ["-O3", f"-I{root_include}"] + (extra_cflags or [])
    if os.path.exists("/usr/local/cuda/include"):
        cpp_cflags.append("-I/usr/local/cuda/include")

    hash_str = hashlib.sha256(
        (cpp_sources + "\n" + cuda_sources + "\n" + "".join(all_cuda_cflags) + "\n" + "".join(cpp_cflags)).encode("utf-8")
    ).hexdigest()[:8]
    cache_key = (name, hash_str)

    if not force_rebuild:
        if cache_key in _LOADED_MODULES:
            return _LOADED_MODULES[cache_key]
        if name in _LOADED_MODULES:
            return _LOADED_MODULES[name]

    try:
        mod_ext_name = f"{name}_{hash_str}"
        mod_cache_dir = Path(_CACHE_DIR) / mod_ext_name
        mod_cache_dir.mkdir(parents=True, exist_ok=True)
        module = load_inline(
            name=mod_ext_name,
            cpp_sources=cpp_sources,
            cuda_sources=cuda_sources,
            functions=functions,
            extra_cuda_cflags=all_cuda_cflags,
            extra_cflags=cpp_cflags,
            verbose=False,
            with_cuda=True,
            build_directory=str(mod_cache_dir),
        )
        _LOADED_MODULES[cache_key] = module
        _LOADED_MODULES[name] = module
        logger.info(f"✅ CUDA C++ module '{mod_ext_name}' loaded successfully")
        return module
    except Exception as e:
        logger.warning(f"❌ CUDA C++ compilation failed for '{name}': {e}")
        _FAILED_MODULES.add(name)
        return None
