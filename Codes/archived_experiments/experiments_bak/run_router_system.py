# #!/usr/bin/env python3
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))

# import pandas as pd
# import numpy as np
# from framework.router.ensemble import RouterEnsemble
# from framework.router.keyword_router import detect_task_type
# from framework.utils.device import get_device

# def main():
#     tasks_config = [
#         {"name": "01_irony", "rank": 32},
#         {"name": "02_negation", "rank": 32},
#         {"name": "03_comparative", "rank": 32}
#     ]
#     data_files = {
#         "01_irony": "./data/01_irony.csv",
#         "02_negation": "./data/02_negation.csv",
#         "03_comparative": "./data/03_comparative.csv"
#     }
#     router = RouterEnsemble(tasks_config, device=get_device())
#     router.load_and_compress()
#     results = router.evaluate_on_full_dataset(data_files)
#     print("🏆 Router Performance on Full Data:")
#     print(f"   Accuracy: {results['accuracy']:.4f}")
#     print(f"   F1 Score: {results['f1']:.4f}")

# if __name__ == "__main__":
#     main()