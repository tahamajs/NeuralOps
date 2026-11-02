#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Lazy public exports for model serialization and deployment bundles."""

from importlib import import_module

_EXPORTS = {
    "export_model": ("exporter", "export_model"),
    "load_exported_model": ("exporter", "load_exported_model"),
    "load_model": ("exporter", "load_model"),
    "compile_to_so": ("exporter", "compile_to_so"),
    "AOTICompiledModelRunner": ("exporter", "AOTICompiledModelRunner"),
    "VLLMAdapter": ("serving", "VLLMAdapter"),
    "TGIAdapter": ("serving", "TGIAdapter"),
    "export_serving_bundle": ("serving", "export_serving_bundle"),
    "validate_serving_metadata": ("serving", "validate_serving_metadata"),
    "SurgeryManifest": ("surgery_manifest", "SurgeryManifest"),
    "replay_surgery_manifest": ("surgery_manifest", "replay_surgery_manifest"),
    "merge_for_inference": ("surgery_manifest", "merge_for_inference"),
    "export_to_hf": ("hf_compat", "export_to_hf"),
    "ExportResult": ("hf_compat", "ExportResult"),
    "export_to_gguf": ("gguf_exporter", "export_to_gguf"),
    "GGUFWriter": ("gguf_exporter", "GGUFWriter"),
    "check_compatibility": ("runtime_compat", "check_compatibility"),
    "CompatibilityReport": ("runtime_compat", "CompatibilityReport"),
    "EngineStatus": ("runtime_compat", "EngineStatus"),
}
_MODULES = {"exporter", "serving", "surgery_manifest", "hf_compat", "runtime_compat", "gguf_exporter"}

__all__ = [*_EXPORTS, *_MODULES]


def __getattr__(name: str):
    """Import export implementation modules only when a public symbol is used."""
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    elif name in _MODULES:
        value = import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
