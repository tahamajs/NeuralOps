"""Abstract BaseRunner architecture for all neuralops benchmark suites."""
from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import torch

from neuralops.benchmarks.config import BenchmarkRunConfig
from neuralops.benchmarks.writers import BenchmarkResult, write_all, write_csv, write_markdown


class BaseRunner(ABC):
    """Abstract base runner providing common lifecycle hooks and result serialization."""

    def __init__(self, config: BenchmarkRunConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        if config.verbose:
            logging.basicConfig(level=logging.DEBUG)

    @abstractmethod
    def run(self) -> list[BenchmarkResult]:
        """Execute benchmark workflow and return list of standard results."""
        ...

    def write_results(self, results: list[BenchmarkResult]) -> Path:
        """Serialize all results to canonical directory hierarchy, CSV, and Markdown."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for res in results:
            write_all(res, out_dir)

        csv_path = out_dir / f"{self.config.dataset}_{self.__class__.__name__.lower()}_summary.csv"
        md_path = out_dir / f"{self.config.dataset}_{self.__class__.__name__.lower()}_summary.md"
        write_csv(results, csv_path)
        write_markdown(results, md_path)
        self.logger.info("Saved summary files: %s and %s", csv_path, md_path)
        return out_dir

    def verify(self) -> bool:
        """Verify device, model, and dataset readiness before executing long sweeps."""
        self.logger.info("Verifying benchmark environment readiness...")
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available. Fallback to CPU.")
            return False
        return True


__all__ = ["BaseRunner"]
