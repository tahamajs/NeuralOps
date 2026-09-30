import torch
from framework.core.triton_kernels.gla_scan import gla_scan_triton
from test_gla_grad import ref_gla_scan

torch.manual_seed(42)
B, H, L, D = 2, 8, 128, 64
device = "cuda"

q = torch.randn(B, H, L, D, device=device, dtype=torch.float32, requires_grad=True)
k = torch.randn(B, H, L, D, device=device, dtype=torch.float32, requires_grad=True)
v = torch.randn(B, H, L, D, device=device, dtype=torch.float32, requires_grad=True)
lam_raw = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32))
lambdas = lam_raw.detach().clone().requires_grad_(True)

q_ref = q.detach().clone().requires_grad_(True)
k_ref = k.detach().clone().requires_grad_(True)
v_ref = v.detach().clone().requires_grad_(True)
lambdas_ref = lam_raw.detach().clone().requires_grad_(True)

y_triton = gla_scan_triton(q, k, v, lambdas)
y_ref = ref_gla_scan(q_ref, k_ref, v_ref, lambdas_ref)

diff = (y_triton - y_ref).abs()
print("fwd abs max: ", diff.max().item())
print("fwd rel norm:", diff.norm().item() / y_ref.norm().item())

grad_out = torch.randn_like(y_triton)
y_triton.backward(grad_out)
y_ref.backward(grad_out)

print("overall q rel norm diff:  ", (q.grad - q_ref.grad).norm().item() / q_ref.grad.norm().item())
print("overall k rel norm diff:  ", (k.grad - k_ref.grad).norm().item() / k_ref.grad.norm().item())
print("overall v rel norm diff:  ", (v.grad - v_ref.grad).norm().item() / v_ref.grad.norm().item())
print("overall lam rel norm diff:", (lambdas.grad - lambdas_ref.grad).norm().item() / lambdas_ref.grad.norm().item())
