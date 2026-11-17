import torch
import torch.nn.functional as F

def gdn2_recurrent_step(
    S: torch.Tensor,
    q_t: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    alpha_t: torch.Tensor,
    beta_erase_t: torch.Tensor,
    beta_write_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decoupled Erase & Write Recurrence step for GDN2.
    """
    kt_norm = F.normalize(k_t, p=2, dim=-1, eps=1e-6)

    inter_mem = torch.matmul(kt_norm, S)
    erase_term = beta_erase_t * torch.matmul(kt_norm.transpose(-1, -2), inter_mem)
    S_decayed = alpha_t * (S - erase_term)

    write_term = beta_write_t * torch.matmul(kt_norm.transpose(-1, -2), v_t)
    S_next = S_decayed + write_term

    qt_norm = F.normalize(q_t, p=2, dim=-1, eps=1e-6)
    y_t = torch.matmul(qt_norm, S_next)
    return y_t, S_next

def gdn2_chunk_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta_erase: torch.Tensor,
    beta_write: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chunk-wise parallel forward execution for GDN2 via exact WY representation.
    """
    B, H, L, d_k = q.shape
    d_v = v.shape[-1]
    device = q.device
    dtype = q.dtype

    if initial_state is None:
        S = torch.zeros((B, H, d_k, d_v), device=device, dtype=torch.float32)
    else:
        S = initial_state.to(device=device, dtype=torch.float32)

    if L == 0:
        return q.new_empty((B, H, 0, d_v)), S.to(dtype)

    c_size = min(max(1, int(chunk_size)), 32)
    pad_len = (c_size - (L % c_size)) % c_size

    if pad_len > 0:
        q_p = F.pad(q, (0, 0, 0, pad_len))
        k_p = F.pad(k, (0, 0, 0, pad_len))
        v_p = F.pad(v, (0, 0, 0, pad_len))
        alpha_p = F.pad(alpha, (0, pad_len), value=1.0)
        be_p = F.pad(beta_erase, (0, pad_len), value=0.0)
        bw_p = F.pad(beta_write, (0, pad_len), value=0.0)
    else:
        q_p, k_p, v_p, alpha_p, be_p, bw_p = q, k, v, alpha, beta_erase, beta_write

    num_chunks = q_p.shape[2] // c_size

    with torch.amp.autocast(device_type=device.type, enabled=False):
        q_blk = F.normalize(q_p.float(), p=2, dim=-1, eps=1e-6).view(B, H, num_chunks, c_size, d_k)
        k_blk = F.normalize(k_p.float(), p=2, dim=-1, eps=1e-6).view(B, H, num_chunks, c_size, d_k)
        v_blk = v_p.float().view(B, H, num_chunks, c_size, d_v)
        a_blk = alpha_p.float().clamp(min=1e-4, max=1.0).view(B, H, num_chunks, c_size)
        be_blk = be_p.float().view(B, H, num_chunks, c_size)
        bw_blk = bw_p.float().view(B, H, num_chunks, c_size)

        gamma = torch.cumsum(torch.log(a_blk), dim=-1)
        exp_gamma = torch.exp(gamma)

        kk = torch.matmul(k_blk, k_blk.transpose(-1, -2))
        g_diff = (gamma.unsqueeze(-1) - gamma.unsqueeze(-2)).clamp(max=0.0)
        tril_strict = torch.tril(torch.ones((c_size, c_size), device=device, dtype=torch.bool), diagonal=-1)
        tril_causal = torch.tril(torch.ones((c_size, c_size), device=device, dtype=torch.bool), diagonal=0)

        exp_diff = torch.exp(g_diff)
        decay_mat = exp_diff * tril_causal.float()
        M = kk * (be_blk.unsqueeze(-1) * (exp_diff * tril_strict.float()))
        I_plus_M = torch.eye(c_size, device=device, dtype=torch.float32) + M

        rhs_v = bw_blk.unsqueeze(-1) * v_blk
        rhs_k = (be_blk * exp_gamma).unsqueeze(-1) * k_blk
        rhs = torch.cat([rhs_v, rhs_k], dim=-1)
        sol = torch.linalg.solve_triangular(I_plus_M, rhs, upper=False, unitriangular=True)
        U_v = sol[..., :d_v]
        W_k = sol[..., d_v:]

        qk_decay = torch.matmul(q_blk, k_blk.transpose(-1, -2)) * decay_mat
        q_decayed = q_blk * exp_gamma.unsqueeze(-1)
        gamma_end = gamma[..., -1:]
        exp_gamma_end = torch.exp(gamma_end).unsqueeze(-1)
        k_to_end = k_blk * torch.exp((gamma_end - gamma).clamp(max=0.0)).unsqueeze(-1)

        outs = []
        for c in range(num_chunks):
            U_c = U_v[:, :, c] - torch.matmul(W_k[:, :, c], S)
            y_c = torch.matmul(q_decayed[:, :, c], S) + torch.matmul(qk_decay[:, :, c], U_c)
            S = exp_gamma_end[:, :, c] * S + torch.matmul(k_to_end[:, :, c].transpose(-1, -2), U_c)
            outs.append(y_c)

        y = torch.cat(outs, dim=2)
    if pad_len > 0:
        y = y[:, :, :L, :]
    if dtype == torch.float16:
        fmax = torch.finfo(dtype).max * 0.5
        y = y.clamp(min=-fmax, max=fmax)
    return y.to(dtype), S.to(dtype)

def householder_reflection(
    v: torch.Tensor,
    u: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Applies Householder orthogonal transformation.
    """
    u_norm_sq = (u * u).sum(dim=-1, keepdim=True) + 1e-6
    dot_uv = (v * u).sum(dim=-1, keepdim=True)
    reflect = 2.0 * beta * dot_uv * (u / u_norm_sq)
    return v - reflect
