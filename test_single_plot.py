import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log = r"D:\SVD_Project\results\unified_adasplash\distilbert\console.log"
out = r"D:\SVD_Project\results\unified_adasplash\distilbert\test_loss.png"

steps, losses = [], []
pat = re.compile(r"fine-tune step (\d+)/\d+ loss=([0-9\.]+)")
if os.path.exists(log):
    with open(log, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))

print(f"Found {len(steps)} steps.")
if steps:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, losses, marker='o', markersize=3, color='#E74C3C', linewidth=1.8)
    ax.set_title("DistilBERT Recovery Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    plt.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close()
    print("Saved test_loss.png successfully.")
