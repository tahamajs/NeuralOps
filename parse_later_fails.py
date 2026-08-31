import re

with open(r"D:\SVD_Project\results\final_paper_run\summary.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    if "FAIL" in line:
        print(line.strip())
