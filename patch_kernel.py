import re, glob
files = ["Codes/framework/attentions/aegis_v4.py", "Codes/framework/attentions/nexus.py", "Codes/framework/attentions/orion.py"]

safe_kernel = """def kernel_linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rff_proj: torch.Tensor,
    use_zero_sum: bool = True,
) -> torch.Tensor:
    B, H, N, D = Q.shape
    F_dim = rff_proj.size(-1)
    orig_dtype = Q.dtype
    
    Q, K, V = Q.float(), K.float(), V.float()
    rff_proj = rff_proj.float()
    
    if use_zero_sum:
        K = K - K.mean(dim=2, keepdim=True)
    Q_norm = torch.nn.functional.normalize(Q, p=2, dim=-1)
    K_norm = torch.nn.functional.normalize(K, p=2, dim=-1)
    
    Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_norm, rff_proj).clamp(max=11.0)
    K_proj = torch.einsum("bhnd,hdf->bhnf", K_norm, rff_proj).clamp(max=11.0)
    Q_feat = torch.exp(Q_proj) / __import__('math').sqrt(F_dim)
    K_feat = torch.exp(K_proj) / __import__('math').sqrt(F_dim)

    KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V)  # (B, H, F, D)
    K_sum = K_feat.sum(dim=2)  # (B, H, F)
    denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1)  # (B, H, N, 1)

    ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV)  # (B, H, N, D)
    ctx = ctx / (denom + 1e-8)
    return ctx.to(orig_dtype)"""

for f in files:
    with open(f, 'r') as file:
        content = file.read()
    
    if "def kernel_linear_attention" not in content: continue
    
    pattern = re.compile(r'def kernel_linear_attention\(.*?->\s*torch\.Tensor:.*?return[^\n]+\n', re.DOTALL)
    new_content = pattern.sub(safe_kernel + "\n", content)
    
    with open(f, 'w') as file:
        file.write(new_content)
    print(f"Patched {f}")
