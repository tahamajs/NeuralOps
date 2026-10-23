#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi. MIT License.
"""Shim for neuralops.layers.svd."""
from __future__ import annotations
from typing import Any
import neuralops.layers.svd as _mod
__getattr__ = _mod.__getattr__
__dir__ = _mod.__dir__
__all__ = list(_mod.__all__)
