import os
import torch

results_dir = r"D:\SVD_Project\results"
count = 0

for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith(".pt"):
            full_path = os.path.join(root, file)
            try:
                # Try loading just metadata/headers
                torch.load(full_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"Removing corrupt checkpoint ({e}): {full_path}")
                try:
                    os.remove(full_path)
                    count += 1
                except Exception as del_e:
                    print(f"Failed to remove {full_path}: {del_e}")

print(f"Cleaned {count} corrupt checkpoint files.")
