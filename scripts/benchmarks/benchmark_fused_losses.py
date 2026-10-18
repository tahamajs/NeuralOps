import torch
import torch.nn.functional as F
import time
from neuralops.core.triton_kernels.fused_ce import FusedLinearCrossEntropy
from neuralops.core.triton_kernels.fused_kd import FusedLinearKLDivergence

device = "cuda"
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Benchmark parameters: B=4, T=2048, D=1024, V=50257 (GPT-2 standard)
B, T, D, V = 4, 2048, 1024, 50257
dtype = torch.float16

print(f"=== BENCHMARK: B={B}, T={T}, D={D}, Vocab={V}, dtype={dtype} ===")
N_tokens = B * T
h = torch.randn(N_tokens, D, dtype=dtype, device=device, requires_grad=True)
w = torch.randn(V, D, dtype=dtype, device=device, requires_grad=True)
b = torch.randn(V, dtype=dtype, device=device, requires_grad=True)
targets = torch.randint(0, V, (N_tokens,), device=device)

# 1. Standard PyTorch Materialized Logits CE
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
logits = F.linear(h, w, b)
loss_std = F.cross_entropy(logits, targets)
loss_std.backward()
torch.cuda.synchronize()
t_std = time.time() - t0
mem_std = torch.cuda.max_memory_allocated() / (1024 ** 2)

del logits, loss_std
h.grad = None
w.grad = None
b.grad = None
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 2. Fused Linear Cross Entropy
fused_ce = FusedLinearCrossEntropy(chunk_size=512)
t0 = time.time()
loss_fused = fused_ce(h, w, targets=targets, bias=b)
loss_fused.backward()
torch.cuda.synchronize()
t_fused = time.time() - t0
mem_fused = torch.cuda.max_memory_allocated() / (1024 ** 2)

print(f"Standard PyTorch CE      - Peak VRAM: {mem_std:8.2f} MB | Latency: {t_std*1000:7.2f} ms")
print(f"FusedLinearCrossEntropy - Peak VRAM: {mem_fused:8.2f} MB | Latency: {t_fused*1000:7.2f} ms")
print(f"CE VRAM Reduction: {(1 - mem_fused/mem_std)*100:.1f}%")

# 3. Knowledge Distillation Benchmark
t_h = torch.randn(N_tokens, D, dtype=dtype, device=device)
t_w = torch.randn(V, D, dtype=dtype, device=device)
t_b = torch.randn(V, dtype=dtype, device=device)

# Standard KD
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
s_logits = F.linear(h, w, b)
with torch.no_grad():
    t_logits = F.linear(t_h, t_w, t_b)
p_s = F.log_softmax(s_logits / 2.0, dim=-1)
p_t = F.softmax(t_logits / 2.0, dim=-1)
loss_kd_std = F.kl_div(p_s, p_t, reduction="batchmean") * 4.0
loss_kd_std.backward()
torch.cuda.synchronize()
t_kd_std = time.time() - t0
mem_kd_std = torch.cuda.max_memory_allocated() / (1024 ** 2)

del s_logits, t_logits, p_s, p_t, loss_kd_std
h.grad = None
w.grad = None
b.grad = None
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Fused Linear KD
fused_kd = FusedLinearKLDivergence(temperature=2.0, chunk_size=512)
t0 = time.time()
loss_kd_fused = fused_kd(h, w, t_h, t_w, student_bias=b, teacher_bias=t_b)
loss_kd_fused.backward()
torch.cuda.synchronize()
t_kd_fused = time.time() - t0
mem_kd_fused = torch.cuda.max_memory_allocated() / (1024 ** 2)

print(f"Standard PyTorch KD      - Peak VRAM: {mem_kd_std:8.2f} MB | Latency: {t_kd_std*1000:7.2f} ms")
print(f"FusedLinearKLDivergence - Peak VRAM: {mem_kd_fused:8.2f} MB | Latency: {t_kd_fused*1000:7.2f} ms")
print(f"KD VRAM Reduction: {(1 - mem_kd_fused/mem_kd_std)*100:.1f}%")

# Larger Vocab benchmark: V=128256 (Llama 3 vocabulary)
print("\n=== LARGE VOCAB BENCHMARK (Llama-3 Vocab): B=2, T=1024, D=1024, Vocab=128256 ===")
B_l, T_l, D_l, V_l = 2, 1024, 1024, 128256
N_l = B_l * T_l
h_l = torch.randn(N_l, D_l, dtype=dtype, device=device, requires_grad=True)
w_l = torch.randn(V_l, D_l, dtype=dtype, device=device, requires_grad=True)
targets_l = torch.randint(0, V_l, (N_l,), device=device)

# Standard CE
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
logits_l = F.linear(h_l, w_l)
loss_std_l = F.cross_entropy(logits_l, targets_l)
loss_std_l.backward()
torch.cuda.synchronize()
t_std_l = time.time() - t0
mem_std_l = torch.cuda.max_memory_allocated() / (1024 ** 2)

del logits_l, loss_std_l
h_l.grad = None
w_l.grad = None
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Fused CE
fused_ce_l = FusedLinearCrossEntropy(chunk_size=256)
t0 = time.time()
loss_fused_l = fused_ce_l(h_l, w_l, targets=targets_l)
loss_fused_l.backward()
torch.cuda.synchronize()
t_fused_l = time.time() - t0
mem_fused_l = torch.cuda.max_memory_allocated() / (1024 ** 2)

print(f"Llama-3 Standard CE      - Peak VRAM: {mem_std_l:8.2f} MB | Latency: {t_std_l*1000:7.2f} ms")
print(f"Llama-3 FusedLinearCrossEntropy - Peak VRAM: {mem_fused_l:8.2f} MB | Latency: {t_fused_l*1000:7.2f} ms")
print(f"Llama-3 CE VRAM Reduction: {(1 - mem_fused_l/mem_std_l)*100:.1f}%")
