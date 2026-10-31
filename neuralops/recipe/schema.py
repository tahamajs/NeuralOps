#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recipe Schema Dataclasses and Fail-Closed Structural Validation.

Implements the EBNF Grammar (Section 3) and Well-Formedness Rules W1, W2, W6, W15.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from neuralops.recipe.errors import (
    E_SCHEMA_INVALID_IDENTIFIER,
    E_SCHEMA_INVALID_SEMVER,
    E_SCHEMA_MISSING_FIELD,
    E_SCHEMA_UNKNOWN_FIELD,
    E_SCHEMA_WRONG_TYPE,
    E_VALIDATION_LEVEL_INVALID,
    RecipeSchemaError,
)

IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
OPERATION_TYPE_RE = re.compile(r"^[a-z]+(\.[a-z0-9_-]+)+$")
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _validate_unknown_fields(data: Dict[str, Any], allowed_keys: set[str], path: str) -> None:
    for k in data.keys():
        if k not in allowed_keys:
            raise RecipeSchemaError(
                f"Unknown field '{k}' not permitted under fail-closed schema policy",
                error_code=E_SCHEMA_UNKNOWN_FIELD,
                path=f"{path}.{k}" if path else k,
            )


@dataclass
class Metadata:
    name: str
    version: str
    description: str
    author: Optional[str] = None
    created_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    license: Optional[str] = None
    production_ready: Optional[bool] = None
    references: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "metadata") -> Metadata:
        if not isinstance(data, dict):
            raise RecipeSchemaError(f"Expected mapping for '{path}', got {type(data).__name__}", E_SCHEMA_WRONG_TYPE, path)

        allowed = {"name", "version", "description", "author", "created_at", "tags", "license", "production_ready", "references"}
        _validate_unknown_fields(data, allowed, path)

        for req in ("name", "version", "description"):
            if req not in data:
                raise RecipeSchemaError(f"Missing required field '{req}' in {path}", E_SCHEMA_MISSING_FIELD, f"{path}.{req}")

        name = data["name"]
        if not isinstance(name, str):
            raise RecipeSchemaError(f"Field '{path}.name' must be string", E_SCHEMA_WRONG_TYPE, f"{path}.name")
        if not IDENTIFIER_RE.match(name):
            raise RecipeSchemaError(f"Identifier '{name}' violates DNS-style naming", E_SCHEMA_INVALID_IDENTIFIER, f"{path}.name")

        ver = data["version"]
        if not isinstance(ver, str):
            raise RecipeSchemaError(f"Field '{path}.version' must be string", E_SCHEMA_WRONG_TYPE, f"{path}.version")
        if not SEMVER_RE.match(ver):
            raise RecipeSchemaError(f"Version '{ver}' violates SemVer 2.0.0", E_SCHEMA_INVALID_SEMVER, f"{path}.version")

        desc = data["description"]
        if not isinstance(desc, str):
            raise RecipeSchemaError(f"Field '{path}.description' must be string", E_SCHEMA_WRONG_TYPE, f"{path}.description")

        tags = data.get("tags", [])
        if not isinstance(tags, list):
            raise RecipeSchemaError(f"Field '{path}.tags' must be a list", E_SCHEMA_WRONG_TYPE, f"{path}.tags")

        return cls(
            name=name,
            version=ver,
            description=desc,
            author=data.get("author"),
            created_at=data.get("created_at"),
            tags=tags,
            license=data.get("license"),
            production_ready=data.get("production_ready"),
            references=data.get("references", []),
        )


@dataclass
class ModelSpec:
    source: str
    dtype: str
    revision: Optional[str] = None
    trust_remote_code: Optional[bool] = None
    device_map: Optional[str] = None
    family: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "model") -> ModelSpec:
        if not isinstance(data, dict):
            raise RecipeSchemaError(f"Expected mapping for '{path}', got {type(data).__name__}", E_SCHEMA_WRONG_TYPE, path)

        allowed = {"source", "dtype", "revision", "trust_remote_code", "device_map", "family"}
        _validate_unknown_fields(data, allowed, path)

        for req in ("source", "dtype"):
            if req not in data:
                raise RecipeSchemaError(f"Missing required field '{req}' in {path}", E_SCHEMA_MISSING_FIELD, f"{path}.{req}")

        source = data["source"]
        if not isinstance(source, str):
            raise RecipeSchemaError(f"Field '{path}.source' must be string", E_SCHEMA_WRONG_TYPE, f"{path}.source")

        dtype = data["dtype"]
        if not isinstance(dtype, str):
            raise RecipeSchemaError(f"Field '{path}.dtype' must be string", E_SCHEMA_WRONG_TYPE, f"{path}.dtype")

        revision = data.get("revision")
        if revision is not None:
            if not isinstance(revision, str):
                raise RecipeSchemaError(f"Field '{path}.revision' must be string", E_SCHEMA_WRONG_TYPE, f"{path}.revision")
            if not GIT_SHA_RE.match(revision):
                raise RecipeSchemaError(
                    f"Model revision '{revision}' must be a full 40-character commit SHA",
                    E_SCHEMA_WRONG_TYPE,
                    f"{path}.revision",
                )

        return cls(
            source=source,
            dtype=dtype,
            revision=revision,
            trust_remote_code=data.get("trust_remote_code"),
            device_map=data.get("device_map"),
            family=data.get("family"),
        )


@dataclass
class Operation:
    type: str
    version: str
    config: Dict[str, Any]
    id: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "operations[i]") -> Operation:
        if not isinstance(data, dict):
            raise RecipeSchemaError(f"Expected mapping for '{path}', got {type(data).__name__}", E_SCHEMA_WRONG_TYPE, path)

        allowed = {"type", "version", "config", "id", "depends_on"}
        _validate_unknown_fields(data, allowed, path)

        for req in ("type", "version", "config"):
            if req not in data:
                raise RecipeSchemaError(f"Missing required field '{req}' in {path}", E_SCHEMA_MISSING_FIELD, f"{path}.{req}")

        op_type = data["type"]
        if not isinstance(op_type, str):
            raise RecipeSchemaError(f"Field '{path}.type' must be string", E_SCHEMA_WRONG_TYPE, f"{path}.type")
        if not OPERATION_TYPE_RE.match(op_type):
            raise RecipeSchemaError(f"Operation type '{op_type}' must be dot-delimited namespace", E_SCHEMA_INVALID_IDENTIFIER, f"{path}.type")

        ver = data["version"]
        if not isinstance(ver, str):
            raise RecipeSchemaError(f"Field '{path}.version' must be string", E_SCHEMA_WRONG_TYPE, f"{path}.version")
        if not SEMVER_RE.match(ver):
            raise RecipeSchemaError(f"Operation version '{ver}' violates SemVer 2.0.0", E_SCHEMA_INVALID_SEMVER, f"{path}.version")

        config = data["config"]
        if not isinstance(config, dict):
            raise RecipeSchemaError(f"Field '{path}.config' must be mapping", E_SCHEMA_WRONG_TYPE, f"{path}.config")

        op_id = data.get("id")
        if op_id is not None:
            if not isinstance(op_id, str):
                raise RecipeSchemaError(f"Field '{path}.id' must be string", E_SCHEMA_WRONG_TYPE, f"{path}.id")
            if not IDENTIFIER_RE.match(op_id):
                raise RecipeSchemaError(f"Operation id '{op_id}' violates identifier format", E_SCHEMA_INVALID_IDENTIFIER, f"{path}.id")

        depends_on = data.get("depends_on", [])
        if not isinstance(depends_on, list):
            raise RecipeSchemaError(f"Field '{path}.depends_on' must be a list", E_SCHEMA_WRONG_TYPE, f"{path}.depends_on")

        return cls(
            type=op_type,
            version=ver,
            config=config,
            id=op_id,
            depends_on=depends_on,
        )


@dataclass
class ValidationSpec:
    profile: str
    policy: str = "fail_fast"
    gates: List[Dict[str, Any]] = field(default_factory=list)
    thresholds: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "validation") -> ValidationSpec:
        if not isinstance(data, dict):
            raise RecipeSchemaError(f"Expected mapping for '{path}', got {type(data).__name__}", E_SCHEMA_WRONG_TYPE, path)

        allowed = {"profile", "policy", "gates", "thresholds"}
        _validate_unknown_fields(data, allowed, path)

        if "profile" not in data:
            raise RecipeSchemaError(f"Missing required field 'profile' in {path}", E_SCHEMA_MISSING_FIELD, f"{path}.profile")

        profile = data["profile"]
        valid_profiles = {"none", "quick", "structural", "standard", "numerical", "thorough", "behavioral", "empirical", "production"}
        if profile not in valid_profiles:
            raise RecipeSchemaError(f"Invalid validation profile '{profile}'", E_SCHEMA_WRONG_TYPE, f"{path}.profile")

        policy = data.get("policy", "fail_fast")
        if policy not in {"fail_fast", "continue_on_warning", "collect_all"}:
            raise RecipeSchemaError(f"Invalid validation policy '{policy}'", E_SCHEMA_WRONG_TYPE, f"{path}.policy")

        return cls(
            profile=profile,
            policy=policy,
            gates=data.get("gates", []),
            thresholds=data.get("thresholds", {}),
        )


@dataclass
class ExecutionSpec:
    device: str = "auto"
    seed: Optional[int] = None
    rollback: bool = True
    memory_budget_gb: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "execution") -> ExecutionSpec:
        if not isinstance(data, dict):
            raise RecipeSchemaError(f"Expected mapping for '{path}', got {type(data).__name__}", E_SCHEMA_WRONG_TYPE, path)

        allowed = {"device", "seed", "rollback", "memory_budget_gb"}
        _validate_unknown_fields(data, allowed, path)

        seed = data.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise RecipeSchemaError(f"Field '{path}.seed' must be an integer", E_SCHEMA_WRONG_TYPE, f"{path}.seed")

        return cls(
            device=data.get("device", "auto"),
            seed=seed,
            rollback=data.get("rollback", True),
            memory_budget_gb=data.get("memory_budget_gb"),
        )


@dataclass
class ArtifactsSpec:
    output_dir: str
    save_model: bool = True
    save_report: bool = True
    format: str = "hf"

    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: str = "artifacts") -> ArtifactsSpec:
        if not isinstance(data, dict):
            raise RecipeSchemaError(f"Expected mapping for '{path}', got {type(data).__name__}", E_SCHEMA_WRONG_TYPE, path)

        allowed = {"output_dir", "save_model", "save_report", "format"}
        _validate_unknown_fields(data, allowed, path)

        if "output_dir" not in data:
            raise RecipeSchemaError(f"Missing required field 'output_dir' in {path}", E_SCHEMA_MISSING_FIELD, f"{path}.output_dir")

        return cls(
            output_dir=data["output_dir"],
            save_model=data.get("save_model", True),
            save_report=data.get("save_report", True),
            format=data.get("format", "hf"),
        )


@dataclass
class Recipe:
    apiVersion: str
    kind: str
    metadata: Metadata
    model: ModelSpec
    operations: List[Operation]
    validation: ValidationSpec = field(default_factory=lambda: ValidationSpec(profile="standard"))
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    artifacts: Optional[ArtifactsSpec] = None

    def validate_schema(self) -> List[str]:
        """Validate well-formedness rules W1, W2, W6, W15 on this instance."""
        errors: List[str] = []
        if self.apiVersion != "neuralops/v1":
            errors.append(f"Invalid apiVersion '{self.apiVersion}', expected 'neuralops/v1'")
        if self.kind != "Recipe":
            errors.append(f"Invalid kind '{self.kind}', expected 'Recipe'")
        if not self.operations:
            errors.append("Operations list must be non-empty")

        # W15: If validation profile is 'none', metadata.production_ready must be explicitly False
        if self.validation.profile == "none" and self.metadata.production_ready is not False:
            errors.append("Rule W15: Profile 'none' requires metadata.production_ready to be explicitly false")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Prune None values from artifacts and execution if not specified
        if self.artifacts is None:
            d.pop("artifacts", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Recipe:
        if not isinstance(data, dict):
            raise RecipeSchemaError(f"Top-level recipe must be a mapping, got {type(data).__name__}", E_SCHEMA_WRONG_TYPE)

        allowed_root = {"apiVersion", "kind", "metadata", "model", "operations", "validation", "execution", "artifacts"}
        _validate_unknown_fields(data, allowed_root, "")

        # W1: Required root fields
        for req in ("apiVersion", "kind", "metadata", "model", "operations"):
            if req not in data:
                raise RecipeSchemaError(f"Rule W1: Missing root field '{req}'", E_SCHEMA_MISSING_FIELD, req)

        api_version = data["apiVersion"]
        if api_version != "neuralops/v1":
            raise RecipeSchemaError(f"Unsupported apiVersion '{api_version}'", E_SCHEMA_WRONG_TYPE, "apiVersion")

        kind = data["kind"]
        if kind != "Recipe":
            raise RecipeSchemaError(f"Unsupported kind '{kind}', expected 'Recipe'", E_SCHEMA_WRONG_TYPE, "kind")

        metadata = Metadata.from_dict(data["metadata"])
        model = ModelSpec.from_dict(data["model"])

        raw_ops = data["operations"]
        if not isinstance(raw_ops, list) or not raw_ops:
            raise RecipeSchemaError("Field 'operations' must be a non-empty list", E_SCHEMA_WRONG_TYPE, "operations")

        operations: List[Operation] = []
        for i, op_data in enumerate(raw_ops):
            operations.append(Operation.from_dict(op_data, path=f"operations[{i}]"))

        validation = ValidationSpec.from_dict(data["validation"]) if "validation" in data else ValidationSpec(profile="standard")

        # W15: Validation profile none requires production_ready: false
        if validation.profile == "none" and metadata.production_ready is not False:
            raise RecipeSchemaError(
                "Rule W15: validation profile 'none' is only permitted when metadata.production_ready is explicitly false",
                E_VALIDATION_LEVEL_INVALID,
                "validation.profile",
            )

        execution = ExecutionSpec.from_dict(data["execution"]) if "execution" in data else ExecutionSpec()
        artifacts = ArtifactsSpec.from_dict(data["artifacts"]) if "artifacts" in data else None

        recipe = cls(
            apiVersion=api_version,
            kind=kind,
            metadata=metadata,
            model=model,
            operations=operations,
            validation=validation,
            execution=execution,
            artifacts=artifacts,
        )
        return recipe
