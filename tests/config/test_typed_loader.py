#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for typed config loading and explicit schema boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from neuralops.config import (
    AblationPipelineConfig,
    AttentionSweepConfig,
    Config,
    DatasetConfig,
    ExperimentConfig,
    QualitySweepConfig,
    load_ablation_pipeline,
    load_attention_sweep_config,
    load_blocklist,
    load_config,
    load_experiment_config,
    load_quality_sweep_config,
)
from neuralops.config.loader import _dict_to_dataclass

CONFIG_DIR = Path("neuralops/config")


def test_nested_dataset_mappings_become_dataclass_instances() -> None:
    @dataclass
    class DatasetCollection:
        datasets: list[DatasetConfig] = field(default_factory=list)

    result = _dict_to_dataclass(
        DatasetCollection,
        {"datasets": [{"name": "sst2", "path": "glue/sst2", "task_type": "classification"}]},
    )

    assert len(result.datasets) == 1
    assert isinstance(result.datasets[0], DatasetConfig)
    assert result.datasets[0].name == "sst2"


def test_integer_fields_reject_fractional_yaml_values() -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        _dict_to_dataclass(ExperimentConfig, {"batch_size": 2.5})


def test_flat_unknown_keys_are_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "flat.yaml"
    path.write_text("model_name: example\nattn_type: full\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown Config field"):
        load_config(path)


def test_default_cli_preset_matches_experiment_schema() -> None:
    config = load_experiment_config(CONFIG_DIR / "default.yaml")

    assert isinstance(config, ExperimentConfig)
    assert config.model_name == "distilbert-base-uncased"
    assert config.task == "sst2"
    assert config.attn_type == "full"
    assert config.compression_rank == 64


def test_attention_sweep_preset_has_typed_lists() -> None:
    config = load_attention_sweep_config(CONFIG_DIR / "sweep_attention_types.yaml")

    assert isinstance(config, AttentionSweepConfig)
    assert config.seeds == [42, 123, 456]
    assert "cluster_attention" in config.attn_types


def test_qwen_quality_preset_has_typed_policy_fields() -> None:
    config = load_quality_sweep_config(CONFIG_DIR / "qwen3_5.yaml")

    assert isinstance(config, QualitySweepConfig)
    assert config.slow_module_timeouts["grecodec"] == 600
    assert "mamba" in config.skip_modules


def test_ablation_stages_are_typed_and_ordered() -> None:
    config = load_ablation_pipeline(CONFIG_DIR / "ablation_pipeline.yaml")

    assert isinstance(config, AblationPipelineConfig)
    assert len(config.stages) == 5
    assert config.stages[0].name == "stage1_naive_svd"
    assert config.stages[0].modules == ["naive_svd"]


def test_blocklist_csv_quotes_and_reasons_are_parsed() -> None:
    entries = load_blocklist(CONFIG_DIR / "blocklist.txt")

    assert "mamba" in entries
    assert entries["mamba"].reason_code == "ARCHITECTURAL_INCOMPATIBLE"
    assert "bidirectional" in entries["mamba"].justification


def test_experiment_cli_schema_and_nested_root_config_are_distinct() -> None:
    assert Config().model.name == "distilbert-base-uncased"
    assert ExperimentConfig().model_name == "distilbert-base-uncased"
