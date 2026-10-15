"""Intelligent results management system package for NeuralOps."""
from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "BenchmarkResult",
    "RunMetadata",
    "ModuleMetadata",
    "ModelMetadata",
    "DatasetMetadata",
    "EnvironmentMetadata",
    "HardwareUsage",
    "BaselineComparison",
    "ReproducibilityInfo",
    "ArtifactPaths",
    "PushInfo",
    "save_all_formats",
    "save_json",
    "save_jsonl_index",
    "save_csv_index",
    "save_markdown_summary",
    "save_parquet_index",
    "compute_result_path",
    "Deduplicator",
    "GitCommitter",
    "HFHubPusher",
    "SummaryGenerator",
    "ResultsIndex",
    "generate_reproduction_command",
    "generate_reproduction_script",
]

_LAZY_RESULTS_MAP: dict[str, tuple[str, str]] = {
    "Deduplicator": ("neuralops.benchmarks.results.dedup", "Deduplicator"),
    "GitCommitter": ("neuralops.benchmarks.results.git_sync", "GitCommitter"),
    "HFHubPusher": ("neuralops.benchmarks.results.hf_sync", "HFHubPusher"),
    "ResultsIndex": ("neuralops.benchmarks.results.index", "ResultsIndex"),
    "generate_reproduction_command": ("neuralops.benchmarks.results.reproduce", "generate_reproduction_command"),
    "generate_reproduction_script": ("neuralops.benchmarks.results.reproduce", "generate_reproduction_script"),
    "compute_result_path": ("neuralops.benchmarks.results.saver", "compute_result_path"),
    "save_all_formats": ("neuralops.benchmarks.results.saver", "save_all_formats"),
    "save_csv_index": ("neuralops.benchmarks.results.saver", "save_csv_index"),
    "save_json": ("neuralops.benchmarks.results.saver", "save_json"),
    "save_jsonl_index": ("neuralops.benchmarks.results.saver", "save_jsonl_index"),
    "save_markdown_summary": ("neuralops.benchmarks.results.saver", "save_markdown_summary"),
    "save_parquet_index": ("neuralops.benchmarks.results.saver", "save_parquet_index"),
    "ArtifactPaths": ("neuralops.benchmarks.results.schema", "ArtifactPaths"),
    "BaselineComparison": ("neuralops.benchmarks.results.schema", "BaselineComparison"),
    "BenchmarkResult": ("neuralops.benchmarks.results.schema", "BenchmarkResult"),
    "DatasetMetadata": ("neuralops.benchmarks.results.schema", "DatasetMetadata"),
    "EnvironmentMetadata": ("neuralops.benchmarks.results.schema", "EnvironmentMetadata"),
    "HardwareUsage": ("neuralops.benchmarks.results.schema", "HardwareUsage"),
    "ModelMetadata": ("neuralops.benchmarks.results.schema", "ModelMetadata"),
    "ModuleMetadata": ("neuralops.benchmarks.results.schema", "ModuleMetadata"),
    "PushInfo": ("neuralops.benchmarks.results.schema", "PushInfo"),
    "ReproducibilityInfo": ("neuralops.benchmarks.results.schema", "ReproducibilityInfo"),
    "RunMetadata": ("neuralops.benchmarks.results.schema", "RunMetadata"),
    "SummaryGenerator": ("neuralops.benchmarks.results.summarizer", "SummaryGenerator"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute loading for neuralops.benchmarks.results symbols."""
    if name in _LAZY_RESULTS_MAP:
        mod_name, attr_name = _LAZY_RESULTS_MAP[name]
        module = importlib.import_module(mod_name)
        val = getattr(module, attr_name)
        globals()[name] = val
        return val

    try:
        module = importlib.import_module(f"neuralops.benchmarks.results.{name}")
        globals()[name] = module
        return module
    except ModuleNotFoundError:
        pass

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
