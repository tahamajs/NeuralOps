import torch
from transformers import AutoModelForSequenceClassification

m = AutoModelForSequenceClassification.from_pretrained("D:/SVD_Project/models/gpt2")
ckpt_path = "D:/SVD_Project/results/gpt2_sst2_lora_baseline/finetuned_model.pt"
try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(ckpt, strict=False)
    print(f"Loaded parent checkpoint: {ckpt_path}")
except Exception as e:
    print(f"Could not load checkpoint: {e}")

print("=" * 100)
print(f"{'Layer Name':45s} | {'Weight Shape':16s} | {'Err@64':8s} | {'Err@128':8s} | {'Fallback@64':11s} | {'Fallback@128':12s}")
print("=" * 100)

max_error = 0.50

for name, mod in m.named_modules():
    if hasattr(mod, "weight") and ("c_attn" in name or "c_proj" in name):
        W = mod.weight.data.t().float() if hasattr(mod, "nf") else mod.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        norm_W = torch.norm(W, p="fro").item()
        
        W_rec64 = U[:, :64] @ torch.diag(S[:64]) @ Vh[:64, :]
        err64 = (torch.norm(W - W_rec64, p="fro") / norm_W).item()
        fb64 = "YES (SKIP)" if err64 > max_error else "NO (COMP)"
        
        W_rec128 = U[:, :128] @ torch.diag(S[:128]) @ Vh[:128, :]
        err128 = (torch.norm(W - W_rec128, p="fro") / norm_W).item()
        fb128 = "YES (SKIP)" if err128 > max_error else "NO (COMP)"
        
        print(f"{name:45s} | {str(list(W.shape)):16s} | {err64:.4f}   | {err128:.4f}    | {fb64:11s} | {fb128:12s}")
