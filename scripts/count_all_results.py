import glob
import os
from pathlib import Path

BASE = Path("D:/SVD_Project") if Path("D:/SVD_Project").exists() else Path(".")
summaries = glob.glob(str(BASE / "results" / "**" / "summary.csv"), recursive=True)
print(f"Total summary.csv files found: {len(summaries)}")

# Count unique modules across summaries
modules = set()
for s in summaries:
    p = Path(s)
    modules.add(p.parent.name)
print(f"Unique parent folder names: {len(modules)}")

# Check distinct benchmark folders in results
top_dirs = set()
for s in summaries:
    rel = Path(s).relative_to(BASE / "results")
    top_dirs.add(rel.parts[0])
print("\nTop-level result suites:")
for td in sorted(top_dirs):
    count = len(glob.glob(str(BASE / "results" / td / "**" / "summary.csv"), recursive=True))
    print(f"  {td:<35}: {count} summaries")
