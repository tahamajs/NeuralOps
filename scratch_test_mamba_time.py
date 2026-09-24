import time
import torch
from transformers import AutoModelForCausalLM
from framework.core.model_mod import inject_module

print("Loading GPT-2 model in fp16...", flush=True)
model = AutoModelForCausalLM.from_pretrained("models/gpt2", torch_dtype=torch.float16).cuda()
print("Injecting Mamba module...", flush=True)
inject_module(model, "mamba")
x = torch.randint(0, 1000, (16, 128), device="cuda")

print("Warming up 1 step...", flush=True)
with torch.no_grad():
    _ = model(x)
torch.cuda.synchronize()

print("Timing 5 steps...", flush=True)
t0 = time.time()
with torch.no_grad():
    for _ in range(5):
        _ = model(x)
torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"5 forwards took: {elapsed:.3f}s ({elapsed/5:.4f}s per forward)", flush=True)
