import subprocess
import sys

with open("D:/svd_project/distilbert_int8.out", "w") as f:
    process = subprocess.Popen(["python", "D:/svd_project/Codes/framework/eval_int8.py", "models/distilbert_sst2", "D:/svd_project/results/unified_a3/distilbert/checkpoints/a3/hf_export/pytorch_model.bin", "sst2", "128"], stdout=f, stderr=subprocess.STDOUT)
    process.communicate()
