import os
import sys

# Ensure low thread memory consumption so MKL/OpenMP does not trigger pagefile alloc limits
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, "D:/svd_project")
sys.path.insert(0, "D:/svd_project/Codes")

from full_a3_sweep_runner import main

if __name__ == "__main__":
    main()
