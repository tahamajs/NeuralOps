"""Shim to allow Pythonic import of SVD-Mamba module."""
import importlib
_mod = importlib.import_module("framework.attentions.SVD-Mamba")
SVDMamba = _mod.SVDMamba
SVDLinear = _mod.SVDLinear
inject_svd_mamba = _mod.inject_svd_mamba
__all__ = ["SVDMamba", "SVDLinear", "inject_svd_mamba"]
