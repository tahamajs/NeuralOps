"""Subpackage exports."""

import importlib

__all__ = []

try:
    _m = importlib.import_module(f".gqa_attention", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except Exception:
    pass

try:
    _m = importlib.import_module(f".mla_attention", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except Exception:
    pass

try:
    _m = importlib.import_module(f".nha_attention", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except Exception:
    pass

try:
    _m = importlib.import_module(f".qcqa_attention", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except Exception:
    pass

try:
    _m = importlib.import_module(f".vig_attention", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except Exception:
    pass

try:
    _m = importlib.import_module(f".wgqa_attention", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except Exception:
    pass
try:
    _m = importlib.import_module(f".dynamic_conv_attention", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type) or callable(_val):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except Exception:
    pass
try:
    _m = importlib.import_module(f".sliding_window", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except Exception:
    pass

try:
    _m = importlib.import_module(f".cross_attention", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except Exception:
    pass

try:
    _m = importlib.import_module(f".mqa", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except Exception:
    pass

try:
    _m = importlib.import_module(f".alibi", __package__)
    for _attr in dir(_m):
        if not _attr.startswith("_"):
            _val = getattr(_m, _attr)
            if isinstance(_val, type):
                globals()[_attr] = _val
                if _attr not in __all__:
                    __all__.append(_attr)
except Exception:
    pass

# Auto-wired submodules
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
