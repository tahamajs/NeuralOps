import re

with open(r"D:\SVD_Project\results\final_paper_run\quality_sweep.log", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "[ERROR]" in line and "QualitySweep: Seed 42 failed for" in line:
        print("".join(lines[max(0, i-5):min(len(lines), i+15)]))
        print("="*60)
