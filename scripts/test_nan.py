import torch
import math
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from neuralops.attentions.nystrom_attention import NystromAttention

model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
orig_attn = model.transformer.h[0].attn
attn = NystromAttention(config=model.config, old_attn=orig_attn).cuda().half()

x = torch.randn(1, 512, 768, device="cuda", dtype=torch.float16)
B, N, D = x.shape
H, d = attn.num_heads, attn.head_dim
scale = 1.0 / math.sqrt(d)

with torch.no_grad():
    Q = attn.query(x).view(B, N, H, d).transpose(1, 2).contiguous()
    K = attn.key(x).view(B, N, attn.num_kv_heads, d).transpose(1, 2).contiguous()
    V = attn.value(x).view(B, N, attn.num_kv_heads, d).transpose(1, 2).contiguous()
    valid_mask = torch.ones((B, N), device="cuda", dtype=torch.bool)
    chunk_size = 256
    m = min(attn.num_landmarks, N)

    # Chunk 0
    c_idx = 0
    start = 0
    end = 256
    C_len = 256
    Q_c = Q[:, :, start:end, :]
    K_c = K[:, :, start:end, :]
    V_c = V[:, :, start:end, :]
    mask_c = valid_mask[:, start:end]

    scores_intra = torch.matmul(Q_c.float(), K_c.float().transpose(-2, -1)) * scale
    causal = torch.tril(torch.ones(C_len, C_len, device="cuda", dtype=torch.bool))
    scores_intra = scores_intra.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)
    attn_intra = F.softmax(scores_intra, dim=-1).half()
    out_intra = torch.matmul(attn_intra, V_c)
    print("out_intra finite:", torch.isfinite(out_intra).all().item())

    # Chunk 1 (c_idx = 1)
    c_idx = 1
    start = 256
    end = 512
    C_len = 256
    Q_c = Q[:, :, start:end, :]
    K_c = K[:, :, start:end, :]
    V_c = V[:, :, start:end, :]
    mask_c = valid_mask[:, start:end]

    # 1. Intra
    scores_intra = torch.matmul(Q_c.float(), K_c.float().transpose(-2, -1)) * scale
    scores_intra = scores_intra.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)
    attn_intra = F.softmax(scores_intra, dim=-1).half()
    out_intra = torch.matmul(attn_intra, V_c)
    print("Chunk 1 out_intra finite:", torch.isfinite(out_intra).all().item())

    # 2. Cross
    K_prev = K[:, :, :start, :]
    V_prev = V[:, :, :start, :]
    mask_prev = valid_mask[:, :start]
    m_prev = min(m, start)

    K_flat = K_prev.reshape(B * H, start, d).transpose(1, 2).contiguous()
    K_landmarks = F.adaptive_avg_pool1d(K_flat, m_prev).transpose(1, 2).contiguous().reshape(B, H, m_prev, d)
    print("K_landmarks finite:", torch.isfinite(K_landmarks).all().item())

    scores1 = torch.matmul(Q_c.float(), K_landmarks.float().transpose(-2, -1)) * scale
    F1 = F.softmax(scores1, dim=-1).half()
    print("F1 finite:", torch.isfinite(F1).all().item())

    Q_flat = Q_c.reshape(B * H, C_len, d).transpose(1, 2).contiguous()
    Q_landmarks = F.adaptive_avg_pool1d(Q_flat, m_prev).transpose(1, 2).contiguous().reshape(B, H, m_prev, d)
    print("Q_landmarks finite:", torch.isfinite(Q_landmarks).all().item())

    scores2 = torch.matmul(Q_landmarks.float(), K_landmarks.float().transpose(-2, -1)) * scale
    F2 = F.softmax(scores2, dim=-1).half()
    print("F2 finite:", torch.isfinite(F2).all().item())

    scores3 = torch.matmul(Q_landmarks.float(), K_prev.float().transpose(-2, -1)) * scale
    F3 = F.softmax(scores3, dim=-1).half()
    print("F3 finite:", torch.isfinite(F3).all().item())

    # F2 pinv
    print("Testing F2_pinv via torch.linalg.pinv...")
    try:
        F2_pinv_svd = torch.linalg.pinv(F2.float()).half()
        print("F2_pinv_svd finite:", torch.isfinite(F2_pinv_svd).all().item())
        print("F2_pinv_svd max:", F2_pinv_svd.abs().max().item())
    except Exception as e:
        print("F2_pinv_svd error:", e)

    print("Testing F2_pinv via iterative pinv...")
    norm_1 = F2.abs().sum(dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
    norm_inf = F2.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
    alpha = 1.0 / (norm_1 * norm_inf + 1e-5)
    V_pinv = alpha * F2.transpose(-2, -1).contiguous()
    I = torch.eye(F2.size(-1), device="cuda", dtype=F2.dtype).unsqueeze(0).unsqueeze(0)
    for it in range(attn.pinv_iterations):
        V_pinv = torch.matmul(V_pinv, 2.0 * I - torch.matmul(F2, V_pinv))
        print(f"  Iter {it} finite: {torch.isfinite(V_pinv).all().item()}, max: {V_pinv.abs().max().item()}")

    step1 = torch.matmul(F3, V_prev)
    print("step1 finite:", torch.isfinite(step1).all().item())

    step2_svd = torch.matmul(F2_pinv_svd, step1)
    print("step2_svd finite:", torch.isfinite(step2_svd).all().item(), "max:", step2_svd.abs().max().item())

    out_cross_svd = torch.matmul(F1, step2_svd)
    print("out_cross_svd finite:", torch.isfinite(out_cross_svd).all().item(), "max:", out_cross_svd.abs().max().item())
