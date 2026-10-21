#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Key-Value (KV) Cache Compression Engines.
"""

import importlib
import logging

__all__ = []

try:
    _m = importlib.import_module(".aatc_xkv_compressor", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except ImportError as exc:
    logging.getLogger(__name__).debug("Optional family module unavailable: %s", exc)

try:
    _m = importlib.import_module(".cachenotes_compressor", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except ImportError as exc:
    logging.getLogger(__name__).debug("Optional family module unavailable: %s", exc)

try:
    _m = importlib.import_module(".delta_kv", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except ImportError as exc:
    logging.getLogger(__name__).debug("Optional family module unavailable: %s", exc)

try:
    _m = importlib.import_module(".hybridkv_compressor", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except ImportError as exc:
    logging.getLogger(__name__).debug("Optional family module unavailable: %s", exc)

try:
    _m = importlib.import_module(".kvtc_compressor", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except ImportError as exc:
    logging.getLogger(__name__).debug("Optional family module unavailable: %s", exc)

try:
    _m = importlib.import_module(".zerokv_neo_compressor", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except ImportError as exc:
    logging.getLogger(__name__).debug("Optional family module unavailable: %s", exc)

from neuralops.compression_engine.kv_cache import aatc_xkv_compressor
from neuralops.compression_engine.kv_cache import cachenotes_compressor
from neuralops.compression_engine.kv_cache import delta_kv
from neuralops.compression_engine.kv_cache import hybridkv_compressor
from neuralops.compression_engine.kv_cache import kvtc_compressor
from neuralops.compression_engine.kv_cache import zerokv_neo_compressor
