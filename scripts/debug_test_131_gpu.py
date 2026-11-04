import torch
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.api import create_attention

ATTENTION_REGISTRY._load_all()
names = list(ATTENTION_REGISTRY._dict.keys())
print("Testing " + str(len(names)) + " attentions with B=1 on GPU...")
failed = []
for idx, name in enumerate(names):
    print("[" + str(idx+1) + "/" + str(len(names)) + "] " + name + "...", flush=True)
    try:
        mod = create_attention(name, dim=64, num_heads=2).to("cuda:0")
        mod.eval()
        x = torch.randn(1, 16, 64, device="cuda:0", dtype=torch.float32)
        with torch.no_grad():
            out = mod(x)
        out_tensor = out[0] if isinstance(out, tuple) else out
        if out_tensor.shape[0] != 1 or torch.isnan(out_tensor).any().item():
            print("  FAILED shape/nan: " + name)
            failed.append(name)
        else:
            print("  OK: " + name)
    except Exception as e:
        print("  EXCEPTION: " + name + " -> " + str(e))
        failed.append(name)

print("FAILED MODULES: " + str(failed))
