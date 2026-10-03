import torch
from framework.core.triton_kernels.gla_scan import gla_scan_triton
from test_gla_grad import ref_gla_scan

B, H, L, D = 2, 4, 32, 16
torch.manual_seed(42)
q = torch.randn(B, H, L, D, device='cuda')
k = torch.randn(B, H, L, D, device='cuda')
v = torch.randn(B, H, L, D, device='cuda')
lam = torch.sigmoid(torch.randn(B, H, L, device='cuda'))

yt = gla_scan_triton(q, k, v, lam)
yr = ref_gla_scan(q, k, v, lam)

diff = (yt - yr).abs()
max_val, max_idx = diff.view(-1).max(0)
print('Max diff:', max_val.item())
b, h, l, d = torch.unravel_index(max_idx, diff.shape)
print(f'At b={b}, h={h}, l={l}, d={d}')
print('yt:', yt[b,h,l,d].item(), 'yr:', yr[b,h,l,d].item())
print('relative error:', (max_val / yr[b,h,l,d].abs()).item())

rel_diff = diff / (yr.abs() + 1e-5)
print('Max relative error:', rel_diff.max().item())
