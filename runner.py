import subprocess
import time
import os
import json

batch_sizes = [64, 128, 256]
results = []

for bs in batch_sizes:
    print(f"Running BS={bs}...")
    
    os.makedirs(f"results/bs_test/bs_{bs}", exist_ok=True)
    
    # Start nvidia-smi logging
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader", "-l", "1"],
        stdout=subprocess.PIPE,
        text=True
    )
    
    start_time = time.time()
    
    cmd = [
        "python", "Codes/framework/run_quality_sweep.py",
        "--modules", "a3",
        "--model", "models/distilbert_sst2",
        "--dataset", "sst2",
        "--batch_size", str(bs),
        "--rank", "64",
        "--max_error", "0.50",
        "--output_dir", f"results/bs_test/bs_{bs}/"
    ]
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        # If there's an OOM or failure, we will capture it
    except Exception as e:
        print("Error:", e)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    smi.terminate()
    smi_out, _ = smi.communicate()
    
    # Parse GPU utilization
    utils = []
    for line in smi_out.splitlines():
        line = line.strip().replace(" %", "")
        if line.isdigit():
            utils.append(int(line))
            
    avg_util = sum(utils) / len(utils) if utils else 0
    
    # Parse accuracy
    acc = 0.0
    json_path = f"results/bs_test/bs_{bs}/distilbert_full_metrics.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
            acc = data.get("performance", {}).get("accuracy_pct", 0.0)
            
    print(f"BS={bs}: util={avg_util:.1f}%, time={elapsed:.1f}s, acc={acc}%")
    results.append({"bs": bs, "util": avg_util, "time": elapsed, "acc": acc})

with open("bs_results.json", "w") as f:
    json.dump(results, f)
print("DONE")
