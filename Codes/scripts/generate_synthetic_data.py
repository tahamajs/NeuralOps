import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.data.synthetic_generator import run_universal_generation

if __name__ == "__main__":
    run_universal_generation(Path.cwd() / "data", total_samples_per_domain=5000)