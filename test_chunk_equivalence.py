import torch
import torch.nn.functional as F

def seq_decay(phi_q, phi_k, v_f, lam, eps=1e-6):
    B, H, N, D_k = phi_q.shape
    D_v = v_f.shape[-1]
    S = torch.zeros(B, H, D_k, D_v, device=phi_q.device)
    Z = torch.zeros(B, H, D_k, 1, device=phi_q.device)
    outputs = []
    for t in range(N):
        qt = phi_q[:, :, t:t+1, :]
        kt = phi_k[:, :, t:t+1, :]
        vt = v_f[:, :, t:t+1, :]
        lam_t = lam[:, :, t] # [B, H, 1, 1]
        kt_t = kt.transpose(-1, -2)
        kv_t = torch.matmul(kt_t, vt)
        S = lam_t * S + kv_t
        Z = lam_t * Z + kt_t
        num_t = torch.matmul(qt, S)
        den_t = torch.matmul(qt, Z) + eps
        outputs.append(num_t / den_t)
    return torch.cat(outputs, dim=2), (S, Z)

def chunk_decay(phi_q, phi_k, v_f, lam, C=64, eps=1e-6):
    B, H, N, D_k = phi_q.shape
    D_v = v_f.shape[-1]
    num_chunks = (N + C - 1) // C
    S_chunk = torch.zeros(B, H, D_k, D_v, device=phi_q.device)
    Z_chunk = torch.zeros(B, H, D_k, 1, device=phi_q.device)
    outputs = []
    
    for c in range(num_chunks):
        start = c * C
        end = min(start + C, N)
        qc = phi_q[:, :, start:end, :] # [B, H, Lc, D_k]
        kc = phi_k[:, :, start:end, :]
        vc = v_f[:, :, start:end, :]
        lam_c = lam[:, :, start:end].squeeze(-1).squeeze(-1) # [B, H, Lc]
        Lc = end - start
        
        # Cumulative decay within chunk
        log_lam_c = torch.log(torch.clamp(lam_c, min=1e-6))
        log_gamma_c = torch.cumsum(log_lam_c, dim=-1) # [B, H, Lc]
        
        # Intra-chunk decay matrix: M_intra[i, j] = exp(log_gamma_c[i] - log_gamma_c[j]) for j <= i
        log_M_intra = log_gamma_c.unsqueeze(-1) - log_gamma_c.unsqueeze(-2) # [B, H, Lc, Lc]
        causal_bool = torch.tril(torch.ones(Lc, Lc, dtype=torch.bool, device=qc.device))
        M_intra = torch.exp(torch.clamp(log_M_intra, max=0.0)) * causal_bool.unsqueeze(0).unsqueeze(0)
        
        aff_intra = torch.matmul(qc, kc.transpose(-1, -2)) * M_intra # [B, H, Lc, Lc]
        num_intra = torch.matmul(aff_intra, vc)
        den_intra = aff_intra.sum(dim=-1, keepdim=True)
        
        # Inter-chunk contribution from S_chunk, Z_chunk
        decay_from_past = torch.exp(torch.clamp(log_gamma_c, max=0.0)).unsqueeze(-1) # [B, H, Lc, 1]
        num_inter = torch.matmul(qc, S_chunk) * decay_from_past # [B, H, Lc, D_v]
        den_inter = torch.matmul(qc, Z_chunk) * decay_from_past # [B, H, Lc, 1]
        
        out_c = (num_intra + num_inter) / (den_intra + den_inter + eps)
        outputs.append(out_c)
        
        # Update S_chunk and Z_chunk to end of this chunk
        total_chunk_decay = torch.exp(torch.clamp(log_gamma_c[:, :, -1:], max=0.0)).unsqueeze(-1) # [B, H, 1, 1]
        log_w_to_end = log_gamma_c[:, :, -1:] - log_gamma_c # [B, H, Lc]
        w_to_end = torch.exp(torch.clamp(log_w_to_end, max=0.0)).unsqueeze(-1) # [B, H, Lc, 1]
        kc_w = kc * w_to_end
        
        S_chunk = total_chunk_decay * S_chunk + torch.matmul(kc_w.transpose(-1, -2), vc)
        Z_chunk = total_chunk_decay * Z_chunk + kc_w.sum(dim=-2, keepdim=True).transpose(-1, -2)
        
    context = torch.cat(outputs, dim=2)
    return context, (S_chunk, Z_chunk)

torch.manual_seed(42)
B, H, N, D_k, D_v = 2, 4, 150, 32, 32
phi_q = torch.rand(B, H, N, D_k)
phi_k = torch.rand(B, H, N, D_k)
v_f = torch.rand(B, H, N, D_v)
lam = torch.rand(B, H, N, 1, 1) * 0.4 + 0.6

out1, (s1, z1) = seq_decay(phi_q, phi_k, v_f, lam)
out2, (s2, z2) = chunk_decay(phi_q, phi_k, v_f, lam, C=32)

print('Out diff:', (out1 - out2).abs().max().item())
print('S diff:', (s1 - s2).abs().max().item())
print('Z diff:', (z1 - z2).abs().max().item())
assert (out1 - out2).abs().max().item() < 1e-4
print('All assertions passed!')
