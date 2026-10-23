#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi. MIT License.
"""Shim for neuralops.layers.fast_kernels."""
from __future__ import annotations
from typing import Any
import neuralops.layers.fast_kernels as _mod
__getattr__ = _mod.__getattr__
__dir__ = _mod.__dir__
__all__ = list(_mod.__all__)
