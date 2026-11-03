#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Transparent In-Memory Import Interceptor for NeuralOps Legacy Subpackages.
Ensures identical module identity (A is B) without maintaining duplicate files.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import types
from importlib.abc import Loader, MetaPathFinder
from types import ModuleType
from typing import Any, Optional, Sequence

logger = logging.getLogger("NeuralOps.CompatImporter")

_RENAMED_ATTENTIONS = {
    'aegis_hybrid': ('hybrid.tri_branch_hybrid_attention', {'AegisHybridAttention': 'TriBranchHybridAttention', 'inject_aegis_hybrid': 'inject_tri_branch_hybrid'}),
    'aegis_spectron_fp8': ('hybrid.spectron_fp8_attention', {'AegisSpectronFP8Attention': 'SpectronFP8Attention', 'inject_aegis_spectron_fp8': 'inject_spectron_fp8'}),
    'aegis_v1': ('hybrid.hybrid_local_kernel_attention', {'AegisAttentionV1': 'HybridLocalKernelAttention'}),
    'aegis_v2': ('hybrid.hybrid_local_kernel_attention', {'AegisAttentionV2': 'HybridLocalKernelAttention', 'inject_aegis_v2': 'inject_hybrid_local_kernel'}),
    'aegis_v3': ('hybrid.hybrid_local_kernel_attention', {'AegisAttentionV3': 'HybridLocalKernelAttention'}),
    'aegis_v4': ('hybrid.hybrid_local_kernel_attention', {'AegisAttentionV4': 'HybridLocalKernelAttention'}),
    'aegis_v5': ('hybrid.split_head_hybrid_attention', {'AegisAttention': 'SplitHeadHybridAttention', 'AegisConfigV5': 'SplitHeadHybridConfig', 'inject_aegis_v5': 'inject_split_head_hybrid'}),
    'aegis_v6': ('hybrid.gated_hybrid_kernel_attention', {'AegisAttentionV3': 'GatedHybridKernelAttention', 'AegisAttentionV6': 'GatedHybridKernelAttention', 'inject_aegis_v6': 'inject_gated_hybrid_kernel'}),
    'leka_v1': ('hybrid.positive_kernel_linear_attention', {'LEKAV1Attention': 'PositiveKernelLinearAttention', 'inject_leka_v1': 'inject_positive_kernel_linear'}),
    'leka_v2': ('hybrid.landmark_local_attention', {'LEKAv2Attention': 'LandmarkLocalAttention', 'inject_leka_v2': 'inject_landmark_local'}),
    'leka_v3': ('hybrid.refined_landmark_attention', {'LEKAv3Attention': 'RefinedLandmarkAttention', 'inject_leka_v3': 'inject_refined_landmark'}),
    'res_svd': ('svd.residual_svd_attention', {'ResSVDAttention': 'ResidualSVDAttention', 'inject_res_svd': 'inject_residual_svd_attention'}),
}


class ExactModuleAliasLoader(Loader):
    """Loader that delegates module resolution directly to canonical module or builds an alias module."""

    def __init__(self, canonical_name: str, is_pkg: bool = False, symbol_aliases: Optional[dict[str, str]] = None):
        self.canonical_name = canonical_name
        self.is_pkg = is_pkg
        self.symbol_aliases = symbol_aliases

    def create_module(self, spec: Any) -> Optional[ModuleType]:
        if not self.symbol_aliases:
            if self.canonical_name not in sys.modules:
                importlib.import_module(self.canonical_name)
            return sys.modules[self.canonical_name]
        
        # Build compatibility alias module with symbol remappings
        canon_mod = importlib.import_module(self.canonical_name)
        alias_mod = types.ModuleType(spec.name)
        alias_mod.__dict__.update(canon_mod.__dict__)
        for alias_sym, canon_sym in self.symbol_aliases.items():
            if hasattr(canon_mod, canon_sym):
                setattr(alias_mod, alias_sym, getattr(canon_mod, canon_sym))
        return alias_mod

    def exec_module(self, module: ModuleType) -> None:
        pass


class NeuralOpsCompatibilityRedirectFinder(MetaPathFinder):
    """Intercepts legacy subpackage requests and resolves them to canonical packages with identity preservation."""

    _EXACT_MAP = {
        "neuralops.attentions": ("neuralops.layers", True),
        "neuralops.compression_engine": ("neuralops.compression", True),
        "neuralops.training": ("neuralops.trainer", True),
        "neuralops.serving": ("neuralops.export.serving", True),
        "neuralops.backward_compat": ("neuralops.core", True),
    }

    _PREFIX_MAP = {
        "neuralops.attentions.": "neuralops.layers.",
        "neuralops.compression_engine.": "neuralops.compression.",
        "neuralops.training.": "neuralops.trainer.",
        "neuralops.serving.": "neuralops.export.serving.",
        "neuralops.backward_compat.attentions.": "neuralops.layers.",
        "neuralops.backward_compat.compression_engine.": "neuralops.compression.",
    }

    def _resolve_canonical(self, canonical_fullname: str) -> Optional[tuple[str, bool, Optional[dict[str, str]]]]:
        # 1. Check renamed attention legacy modules
        parts = canonical_fullname.split(".")
        if len(parts) == 3 and parts[0] == "neuralops" and parts[1] == "layers":
            leaf = parts[2]
            if leaf in _RENAMED_ATTENTIONS:
                rel_path, sym_map = _RENAMED_ATTENTIONS[leaf]
                target_canon = f"neuralops.layers.{rel_path}"
                return (target_canon, False, sym_map)

        # 2. Try direct import
        try:
            c_spec = importlib.util.find_spec(canonical_fullname)
            if c_spec is not None:
                is_pkg = c_spec.submodule_search_locations is not None
                return (canonical_fullname, is_pkg, None)
        except Exception:
            pass

        # 3. If flat submodule lookup failed in layers or compression, try searching subdirectories
        if len(parts) == 3 and parts[0] == "neuralops":
            pkg, mod = parts[1], parts[2]
            subpkgs = []
            if pkg == "layers":
                subpkgs = ['linear', 'svd', 'sparse', 'gated', 'dense', 'ssm', 'hybrid', 'modern', 'adaptive', 'fast_kernels', 'experimental']
            elif pkg == "compression":
                subpkgs = ['svd', 'quantization', 'pruning', 'low_rank', 'tensor_decomp', 'kv_cache', 'experimental']
            
            for sub in subpkgs:
                candidate = f"neuralops.{pkg}.{sub}.{mod}"
                try:
                    c_spec = importlib.util.find_spec(candidate)
                    if c_spec is not None:
                        is_pkg = c_spec.submodule_search_locations is not None
                        return (candidate, is_pkg, None)
                except Exception:
                    continue
        return None

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType] = None,
    ) -> Optional[Any]:
        # 1. Exact package root matching
        if fullname in self._EXACT_MAP:
            target_name, is_pkg = self._EXACT_MAP[fullname]
            loader = ExactModuleAliasLoader(target_name, is_pkg=is_pkg)
            return importlib.util.spec_from_loader(fullname, loader, is_package=is_pkg)

        # 2. Submodule prefix matching
        for old_prefix, new_prefix in self._PREFIX_MAP.items():
            if fullname.startswith(old_prefix):
                subpath = fullname[len(old_prefix):]
                canonical_fullname = f"{new_prefix}{subpath}"
                resolved = self._resolve_canonical(canonical_fullname)
                if resolved is not None:
                    canon, is_pkg, sym_map = resolved
                    loader = ExactModuleAliasLoader(canon, is_pkg=is_pkg, symbol_aliases=sym_map)
                    return importlib.util.spec_from_loader(fullname, loader, is_package=is_pkg)
        return None


def install_compatibility_interceptor() -> None:
    """Installs the compatibility interceptor at the front of sys.meta_path idempotently."""
    if not any(isinstance(finder, NeuralOpsCompatibilityRedirectFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, NeuralOpsCompatibilityRedirectFinder())
        logger.debug("NeuralOpsCompatibilityRedirectFinder active in sys.meta_path.")
