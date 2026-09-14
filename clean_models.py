import os

results_dir = r"D:\svd_project\results"
removed = 0
freed = 0
for root, dirs, files in os.walk(results_dir):
    if "baseline" in root.lower():
        continue
    for f in files:
        if f.endswith((".pt", ".bin", ".safetensors")):
            p = os.path.join(root, f)
            try:
                sz = os.path.getsize(p)
                os.remove(p)
                removed += 1
                freed += sz
            except Exception as e:
                pass
print(f"Deleted {removed} files, freed {freed / (1024*1024):.2f} MB")
