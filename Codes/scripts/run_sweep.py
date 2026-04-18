import subprocess
import itertools
from datetime import datetime

models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "albert-base-v2"]
ranks = [64, 128, 256]
tasks = ["cola", "sst2", "mrpc", "qqp", "qnli", "rte", "mnli", "stsb"]

for model, rank, task in itertools.product(models, ranks, tasks):
    run_name = f"{model.replace('/', '_')}_svd_{rank}_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cmd = [
        "python", "experiments/run_experiment.py",
        "--model", model,
        "--attention", "svd",
        "--rank", str(rank),
        "--task", task,
        "--run_name", run_name
    ]
    print(" ".join(cmd))
    subprocess.run(cmd)