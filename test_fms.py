import torch, gc
from framework.attentions.factorised_matrix_state import FactorisedMatrixState

m = FactorisedMatrixState(hidden_size=128, num_attention_heads=4).cuda().half()
for bs in [32, 64, 128, 256]:
    try:
        torch.cuda.empty_cache()
        gc.collect()
        x = torch.randn(bs, 192, 128, device='cuda', dtype=torch.float16, requires_grad=True)
        out, _ = m(x)
        loss = out.sum()
        loss.backward()
        mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Batch {bs}: PASS, Peak VRAM: {mem:.1f} MB")
    except Exception as e:
        print(f"Batch {bs}: FAIL ({e})")
