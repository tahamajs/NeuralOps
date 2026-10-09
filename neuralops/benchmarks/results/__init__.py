"""Intelligent results management system package for NeuralOps."""

from neuralops.benchmarks.results.dedup import Deduplicator
from neuralops.benchmarks.results.git_sync import GitCommitter
from neuralops.benchmarks.results.hf_sync import HFHubPusher
from neuralops.benchmarks.results.index import ResultsIndex
from neuralops.benchmarks.results.reproduce import (
    generate_reproduction_command,
    generate_reproduction_script,
)
from neuralops.benchmarks.results.saver import (
    compute_result_path,
    save_all_formats,
    save_csv_index,
    save_json,
    save_jsonl_index,
    save_markdown_summary,
    save_parquet_index,
)
from neuralops.benchmarks.results.schema import (
    ArtifactPaths,
    BaselineComparison,
    BenchmarkResult,
    DatasetMetadata,
    EnvironmentMetadata,
    HardwareUsage,
    ModelMetadata,
    ModuleMetadata,
    PushInfo,
    ReproducibilityInfo,
    RunMetadata,
)
from neuralops.benchmarks.results.summarizer import SummaryGenerator

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
