#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Shared lazy-export support for attention and compression families."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def _project_import_target(package_name: str, node: ast.ImportFrom) -> str | None:
    """Resolve an import-from target when it points into NeuralOps."""
    if node.level:
        relative_name = "." * node.level + (node.module or "")
        try:
            target = importlib.util.resolve_name(relative_name, package_name)
        except ImportError:
            return None
    else:
        target = node.module or ""
    return target if target.startswith("neuralops.") else None


def _scan_family(namespace: dict[str, Any]) -> tuple[dict[str, tuple[str, str]], tuple[str, ...]]:
    """Build symbol metadata from source without importing implementations."""
    package_name = namespace["__name__"]
    package_dir = Path(namespace["__file__"]).parent
    exports: dict[str, tuple[str, str]] = {}
    module_names: list[str] = []

    for source_path in sorted(package_dir.glob("*.py")):
        module_name = source_path.stem
        if module_name == "__init__" or module_name.startswith("_"):
            continue

        module_names.append(module_name)
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue

        module_path = f"{package_name}.{module_name}"
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                exports[node.name] = (module_path, node.name)
            elif isinstance(node, ast.ImportFrom):
                target = _project_import_target(module_path, node)
                if target is None:
                    continue
                for alias in node.names:
                    public_name = alias.asname or alias.name
                    if (
                        public_name[:1].isupper()
                        and any(character.islower() for character in public_name)
                        and not public_name.startswith("_")
                    ):
                        exports[public_name] = (target, alias.name)

    return exports, tuple(module_names)


def install_lazy_family(namespace: dict[str, Any]) -> list[str]:
    """Install module-level lazy exports while preserving family imports."""
    exports, module_names = _scan_family(namespace)
    package_name = namespace["__name__"]
    module_map = {name: f"{package_name}.{name}" for name in module_names}
    namespace["_LAZY_EXPORTS"] = exports
    namespace["_FAMILY_MODULES"] = module_map

    def __getattr__(name: str) -> Any:
        if name in module_map:
            value = importlib.import_module(module_map[name])
        elif name in exports:
            module_path, attribute_name = exports[name]
            module: ModuleType = importlib.import_module(module_path)
            try:
                value = getattr(module, attribute_name)
            except AttributeError as exc:
                raise AttributeError(
                    f"module {package_name!r} has no attribute {name!r}"
                ) from exc
        else:
            raise AttributeError(f"module {package_name!r} has no attribute {name!r}")
        namespace[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted(set(namespace) | set(exports) | set(module_map))

    namespace["__getattr__"] = __getattr__
    namespace["__dir__"] = __dir__
    return sorted(exports)
