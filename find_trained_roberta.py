from pathlib import Path
import os

print("=== Scanning D:\\svd_project for roberta weights ===")
for root, dirs, files in os.walk(r"D:\svd_project"):
    for f in files:
        fl = f.lower()
        if any(ext in fl for ext in [".pt", ".bin", ".safetensors"]):
            if "roberta" in root.lower() or "roberta" in fl:
                p = os.path.join(root, f)
                try:
                    sz = os.path.getsize(p)
                    print(f"{p} ({sz / (1024*1024):.2f} MB)")
                except Exception as e:
                    print(f"{p} (error: {e})")

print("\n=== Scanning D:\\models if exists ===")
if os.path.exists(r"D:\models"):
    for root, dirs, files in os.walk(r"D:\models"):
        for f in files:
            p = os.path.join(root, f)
            print(f"{p} ({os.path.getsize(p) / (1024*1024):.2f} MB)")

print("\n=== Scanning HuggingFace cache ===")
hf_cache = os.path.expanduser(r"~/.cache/huggingface/hub")
if os.path.exists(hf_cache):
    for d in os.listdir(hf_cache):
        if "roberta" in d.lower():
            print(f"HF cache: {d}")
