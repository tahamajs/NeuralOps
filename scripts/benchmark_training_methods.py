import torch
import time
import torch.nn as nn
from neuralops.trainer.optimizers import Lion, SophiaG, AdaFactor, Muon, get_optimizer_memory_footprint
from neuralops.trainer import Unified6PillarDistillationLoss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Benchmarking on device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

# 1. Benchmark Unified 6-Pillar Distillation Loss
B, S, V, D = 4, 512, 16000, 768
s_logits = torch.randn(B, S, V, device=device, requires_grad=True)
t_logits = torch.randn(B, S, V, device=device)
s_hid = torch.randn(B, S, D, device=device, requires_grad=True)
t_hid = torch.randn(B, S, D, device=device)
s_val = torch.randn(B, 12, S, 64, device=device, requires_grad=True)
t_val = torch.randn(B, 12, S, 64, device=device)
W = torch.randn(D, D, device=device, requires_grad=True)


loss_fn = Unified6PillarDistillationLoss()

# Warmup
for _ in range(3):
    out = loss_fn(s_logits, t_logits, s_hid, t_hid, s_val, t_val, [W])
    out["loss_total"].backward()

if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.perf_counter()
iters = 10
for _ in range(iters):
    s_logits.grad = None
    s_hid.grad = None
    s_val.grad = None
    W.grad = None
    out = loss_fn(s_logits, t_logits, s_hid, t_hid, s_val, t_val, [W])
    out["loss_total"].backward()
if torch.cuda.is_available():
    torch.cuda.synchronize()
dur = (time.perf_counter() - start) / iters
tokens_per_sec = (B * S) / dur
print(f"Unified 6-Pillar Loss step latency: {dur*1000:.2f} ms ({tokens_per_sec:.0f} tokens/s)")

# 2. Benchmark Modern Optimizers Memory & Step Latency
model = nn.Sequential(nn.Linear(2048, 2048, bias=False), nn.Linear(2048, 2048, bias=False)).to(device)
for name, opt in [("AdamW", torch.optim.AdamW(model.parameters(), lr=1e-3)),
                 ("Lion", Lion(model.parameters(), lr=1e-4)),
                 ("SophiaG", SophiaG(model.parameters(), lr=1e-4)),
                 ("AdaFactor", AdaFactor(model.parameters(), lr=1e-3)),
                 ("Muon", Muon(model.parameters(), lr=0.02))]:
    # Warmup
    x = torch.randn(16, 2048, device=device)
    loss = model(x).sum()
    loss.backward()
    if hasattr(opt, "update_hessian"): opt.update_hessian()
    opt.step()
    opt.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        loss = model(x).sum()
        loss.backward()
        if hasattr(opt, "update_hessian"): opt.update_hessian()
        opt.step()
        opt.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    step_time = (time.perf_counter() - t0) / 20.0
    mem = get_optimizer_memory_footprint(opt)
    print(f"Optimizer {name:10s} | Step: {step_time*1000:.3f} ms | State VRAM: {mem['total_mb']:.2f} MB")
