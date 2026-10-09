"""Migration script to convert existing benchmark JSON results to canonical 15-section schema."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from neuralops.benchmarks.results.saver import save_all_formats
from neuralops.benchmarks.results.schema import BenchmarkResult
from neuralops.benchmarks.results.summarizer import SummaryGenerator

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)


def migrate_results(results_dir: Path | str = "results") -> dict[str, int]:
    """Scan and migrate legacy JSON results to canonical schema and all 5 formats."""
    r_dir = Path(results_dir)
    if not r_dir.exists():
        LOG.error("Directory %s does not exist.", r_dir)
        return {"migrated": 0, "skipped": 0, "errors": 0}

    migrated_count = 0
    skipped_count = 0
    error_count = 0

    all_json_files = list(r_dir.rglob("*.json"))
    LOG.info("Found %d JSON files to inspect in %s", len(all_json_files), r_dir)

    for p in all_json_files:
        if p.name in ("run_index.json", "run_state.json", "package.json", "REORGANIZATION_MANIFEST.json", "RESULTS_MASTER_INDEX.json"):
            skipped_count += 1
            continue

        try:
            raw_text = p.read_text(encoding="utf-8")
            data = json.loads(raw_text)
            if not isinstance(data, dict):
                skipped_count += 1
                continue

            # Check if this is a benchmark result
            if "module" not in data:
                skipped_count += 1
                continue

            # Load via schema with legacy compatibility
            benchmark_res = BenchmarkResult.from_dict(data)

            # Re-save into canonical schema and all formats
            save_all_formats(benchmark_res, output_dir=r_dir)
            migrated_count += 1
        except Exception as e:
            LOG.warning("Could not migrate %s: %s", p, e)
            error_count += 1

    LOG.info("Migration complete: %d migrated, %d skipped, %d errors", migrated_count, skipped_count, error_count)

    # Regenerate summaries
    try:
        summarizer = SummaryGenerator(output_dir=r_dir)
        summarizer.generate_all()
        LOG.info("Successfully regenerated rolling summaries.")
    except Exception as e:
        LOG.warning("Could not regenerate summaries post-migration: %s", e)

    return {"migrated": migrated_count, "skipped": skipped_count, "errors": error_count}


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy NeuralOps results to canonical 15-section schema.")
    parser.add_argument("--results-dir", type=str, default="results", help="Path to results directory")
    args = parser.parse_args()
    migrate_results(args.results_dir)


if __name__ == "__main__":
    main()
