import sys
import time
from pathlib import Path

# Add paths
proj_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(proj_dir))
sys.path.insert(0, str(proj_dir / "Codes"))

import torch
from transformers import AutoModelForSequenceClassification

print("=== SYSTEM & PYTORCH INFO ===")
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    print("Device capability:", torch.cuda.get_device_capability(0))
    print("Allocated memory (MB):", torch.cuda.memory_allocated(0) / 1024**2)
    print("Reserved memory (MB):", torch.cuda.memory_reserved(0) / 1024**2)

print("\n=== MODEL TIMING TEST ===")
local_candidates = [
    proj_dir / "models" / "distilbert_sst2",
    Path("D:/SVD_Project/models/distilbert_sst2"),
]
ckpt = None
for c in local_candidates:
    if c.exists() and ((c / "model.safetensors").exists() or (c / "pytorch_model.bin").exists()):
        ckpt = str(c)
        break
if ckpt is None:
    ckpt = "distilbert-base-uncased-finetuned-sst-2-english"

print(f"Loading checkpoint from: {ckpt}")
t_load0 = time.time()
m = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2).to("cuda")
print(f"Model loaded in {time.time() - t_load0:.2f}s")
print("Model device:", next(m.parameters()).device)

x = torch.randint(0, 1000, (64, 128)).cuda()
labels = torch.randint(0, 2, (64,)).cuda()

# Warmup
out = m(x, labels=labels).loss
out.backward()
torch.cuda.synchronize()

print("\n=== RUNNING 50 FWD+BWD STEPS (BATCH 64, SEQ 128) ===")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    m.zero_grad()
    out = m(x, labels=labels).loss
    out.backward()
torch.cuda.synchronize()
t_total = time.time() - t0
print(f"50 fwd+bwd steps took: {t_total:.4f} seconds ({t_total/50*1000.0:.2f} ms/step)")
print("Final allocated VRAM (MB):", torch.cuda.memory_allocated(0) / 1024**2)
print("Peak allocated VRAM (MB):", torch.cuda.max_memory_allocated(0) / 1024**2)
