#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
SVD Factorized Attention Mechanisms.
"""

import importlib
import logging

__all__ = []

try:
    _m = importlib.import_module(".adaptive_svd_attention", __package__)
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
    _m = importlib.import_module(".asvd_attention", __package__)
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
    _m = importlib.import_module(".compressed_attention", __package__)
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
    _m = importlib.import_module(".compressed_mha_attention", __package__)
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
    _m = importlib.import_module(".factorised_matrix_state", __package__)
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
    _m = importlib.import_module(".fused_svd_kernel", __package__)
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
    _m = importlib.import_module(".lora_attention", __package__)
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
    _m = importlib.import_module(".minimal_nova", __package__)
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
    _m = importlib.import_module(".minimal_svd", __package__)
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
    _m = importlib.import_module(".naive_svd", __package__)
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
    _m = importlib.import_module(".projection_factorized_attn", __package__)
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
    _m = importlib.import_module(".residual_svd_attention", __package__)
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
    _m = importlib.import_module(".salt_attention", __package__)
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
    _m = importlib.import_module(".svd_attention", __package__)
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
    _m = importlib.import_module(".svd_compressed", __package__)
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
    _m = importlib.import_module(".svd_linear_attention", __package__)
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
    _m = importlib.import_module(".svd_linear_decay", __package__)
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
    _m = importlib.import_module(".svd_llm_v2_attention", __package__)
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
    _m = importlib.import_module(".svd_mamba", __package__)
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
    _m = importlib.import_module(".svd_projection_attention", __package__)
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
    _m = importlib.import_module(".triton_svd_linear_attention", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except ImportError as exc:
    logging.getLogger(__name__).debug("Optional family module unavailable: %s", exc)

from neuralops.attentions.svd import adaptive_svd_attention
from neuralops.attentions.svd import asvd_attention
from neuralops.attentions.svd import compressed_attention
from neuralops.attentions.svd import compressed_mha_attention
from neuralops.attentions.svd import factorised_matrix_state
from neuralops.attentions.svd import fused_svd_kernel
from neuralops.attentions.svd import lora_attention
from neuralops.attentions.svd import minimal_nova
from neuralops.attentions.svd import minimal_svd
from neuralops.attentions.svd import naive_svd
from neuralops.attentions.svd import projection_factorized_attn
from neuralops.attentions.svd import residual_svd_attention
from neuralops.attentions.svd import salt_attention
from neuralops.attentions.svd import svd_attention
from neuralops.attentions.svd import svd_compressed
from neuralops.attentions.svd import svd_linear_attention
from neuralops.attentions.svd import svd_linear_decay
from neuralops.attentions.svd import svd_llm_v2_attention
from neuralops.attentions.svd import svd_mamba
from neuralops.attentions.svd import svd_projection_attention
from neuralops.attentions.svd import triton_svd_linear_attention
