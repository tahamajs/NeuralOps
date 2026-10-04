import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp


def _scan_chunk(h_in, da_c, dbu_c, c_c):
    ys = []
    curr = h_in
    for i in range(da_c.shape[2]):
        curr = da_c[:, :, i, :] * curr + dbu_c[:, :, i, :]
        c_t = c_c[:, :, i].unsqueeze(-1)
        yt = torch.bmm(curr, c_t).squeeze(-1)
        ys.append(yt)
    return torch.stack(ys, dim=-1), curr

def test_checkpointed_backward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    B, H, L, D_s = 1, 1536, 256, 16
    h_init = torch.zeros(B, H, D_s, device=device, dtype=dtype)
    da = torch.randn(B, H, L, D_s, device=device, dtype=dtype, requires_grad=True)
    dbu = torch.randn(B, H, L, D_s, device=device, dtype=dtype, requires_grad=True)
    C = torch.randn(B, D_s, L, device=device, dtype=dtype, requires_grad=True)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    chunk_size = 32
    ys_chunked = []
    curr_chunk = h_init
    for s in range(0, L, chunk_size):
        e = min(s + chunk_size, L)
        y_c, curr_chunk = cp.checkpoint(
            _scan_chunk,
            curr_chunk,
            da[:, :, s:e, :],
            dbu[:, :, s:e, :],
            C[:, :, s:e],
            use_reentrant=False,
        )
        ys_chunked.append(y_c)
    y_chunked = torch.cat(ys_chunked, dim=-1)

    loss = y_chunked.sum()
    loss.backward()

    print("Checkpointed Backward: SUCCESS!")
    print(f"da.grad shape: {da.grad.shape}, norm: {da.grad.norm().item():.4f}")
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak VRAM for 1 layer: {peak_mb:.2f} MB")

if __name__ == "__main__":
    test_checkpointed_backward()
