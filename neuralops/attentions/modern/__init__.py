#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Modern Attention Mechanisms.
"""

import importlib
import logging

__all__ = []

try:
    _m = importlib.import_module(".gqa_attention", __package__)
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
    _m = importlib.import_module(".mla_attention", __package__)
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
    _m = importlib.import_module(".nha_attention", __package__)
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
    _m = importlib.import_module(".qcqa_attention", __package__)
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
    _m = importlib.import_module(".vig_attention", __package__)
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
    _m = importlib.import_module(".wgqa_attention", __package__)
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
    _m = importlib.import_module(".dynamic_conv_attention", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type) or callable(_val):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except ImportError as exc:
    logging.getLogger(__name__).debug("Optional family module unavailable: %s", exc)
try:
    _m = importlib.import_module(".sliding_window", __package__)
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
    _m = importlib.import_module(".cross_attention", __package__)
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
    _m = importlib.import_module(".mqa", __package__)
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
    _m = importlib.import_module(".alibi", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except ImportError as exc:
    logging.getLogger(__name__).debug("Optional family module unavailable: %s", exc)

from neuralops.attentions.modern import alibi
from neuralops.attentions.modern import cross_attention
from neuralops.attentions.modern import dynamic_conv_attention
from neuralops.attentions.modern import gqa_attention
from neuralops.attentions.modern import mla_attention
from neuralops.attentions.modern import mqa
from neuralops.attentions.modern import nha_attention
from neuralops.attentions.modern import qcqa_attention
from neuralops.attentions.modern import unified_rope
from neuralops.attentions.modern import vig_attention
from neuralops.attentions.modern import wgqa_attention
