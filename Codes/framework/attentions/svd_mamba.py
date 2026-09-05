"""Shim to allow Pythonic import of SVD-Mamba module."""
import importlib
_mod = importlib.import_module("framework.attentions.SVD-Mamba")
SVDMamba = _mod.SVDMamba
SelectiveSSM = _mod.SelectiveSSM
inject_svd_mamba = _mod.inject_svd_mamba
__all__ = ["SVDMamba", "SelectiveSSM", "inject_svd_mamba"]
