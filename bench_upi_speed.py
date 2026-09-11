import time
import types
import torch
from framework.attentions.upi_stuffed_mamba import UPIMambaBlock

configs = [
    ("DistilBERT (Bidirectional)", False, 768),
    ("RoBERTa (Bidirectional)", False, 768),
    ("GPT-2 (Causal Decoder)", True, 768),
    ("Qwen (Causal GQA)", True, 1024),
]

print("=" * 65)
print(f"{'Model Architecture':28s} | {'Latency (L=512)':16s} | {'Peak VRAM':12s}")
print("=" * 65)

for name, is_dec, d in configs:
    cfg = types.SimpleNamespace(hidden_size=d, is_decoder=is_dec)
    m = UPIMambaBlock(cfg, dim=d, d_state=16).cuda().half().eval()
    x = torch.randn(2, 512, d, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(10):
        m(x)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        m(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    lat = (t1 - t0) / 50 * 1000
    vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"{name:28s} | {lat:6.2f} ms         | {vram:6.1f} MB")

print("=" * 65)
