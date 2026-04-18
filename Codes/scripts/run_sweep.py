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



    import subprocess
import sys
import argparse

RANKS = [32, 48, 64, 96, 128]

def run(skip_coreml=False):
    for r in RANKS:
        print(f"\n=== Processing Rank {r} ===")
        subprocess.run([sys.executable, "-m", "soa.export_onnx", "--rank", str(r), "--onnx_path", f"soa_bert_rank_{r}.onnx"], check=False)
        if not skip_coreml:
            subprocess.run([sys.executable, "-m", "soa.export_coreml", "--onnx_path", f"soa_bert_rank_{r}.onnx", "--mlmodel_path", f"soa_bert_rank_{r}.mlmodel"], check=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_coreml", action="store_true")
    args = parser.parse_args()
    run(skip_coreml=args.skip_coreml)