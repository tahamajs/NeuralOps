import os
filepath = r".gitignore"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

if "results/*.log" not in content:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\nresults/*.log\n")
