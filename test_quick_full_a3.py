import sys
import time
from pathlib import Path

proj_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(proj_dir))
sys.path.insert(0, str(proj_dir / "Codes"))

from framework.run_quality_sweep import run_quality_benchmark_sweep

print("=== STARTING 2-MODULE TEST: full, a3 ===")
t0 = time.time()
run_quality_benchmark_sweep(
    modules=["full", "a3"],
    models=["distilbert"],
    datasets=["sst2"],
    seeds=[42],
    epochs=1,
    max_steps=100,  # 100 fast steps for verification test
    learning_rate=3e-5,
    weight_decay=0.01,
    batch_size=64,
    seq_len=128,
    timeout=300,
    output_dir="results/test_quick_full_a3",
    pretrained=True,
    gradient_checkpointing=True,
)
print(f"=== TEST COMPLETED in {time.time() - t0:.2f} seconds ===")
