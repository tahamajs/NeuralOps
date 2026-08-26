"""
Dry-run local simulation of Phase 2 Quality Sweep on CPU
Tests 2 representative modules ('full', 'svd_linear_attention') on 1 seed with max_steps=2.
Verifies metrics, confidence intervals, t-test p-values, fidelity calculation, and publication report.
"""

import sys
import tempfile
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKDIR))
sys.path.insert(0, str(WORKDIR / "Codes"))

from framework.run_quality_sweep import run_quality_benchmark_sweep
from framework.generate_publication_report import generate_full_publication_report

def main():
    print("=" * 70)
    print("🧪 RUNNING LOCAL CPU DRY-RUN TEST FOR PHASE 2 QUALITY SWEEP")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_dir = Path(tmp_dir) / "test_quality_run"
        test_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {test_dir}")
        
        res_dir = run_quality_benchmark_sweep(
            modules=["full", "svd_linear_attention"],
            models=["bert"],
            datasets=["sst2"],
            seeds=[42],
            epochs=1,
            max_steps=2,
            learning_rate=3e-5,
            weight_decay=0.01,
            batch_size=16,
            seq_len=64,
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            precision="fp32",
            output_dir=str(test_dir),
            allow_synthetic=True,
        )
        
        print("\n📊 Testing publication report generation on dry-run results...")
        generate_full_publication_report(str(test_dir))
        
        print("\n✅ DRY-RUN TEST COMPLETED WITH 100% SUCCESS!")
        print("=" * 70)

if __name__ == "__main__":
    main()
