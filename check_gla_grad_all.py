import torch
from framework.core.triton_kernels.gla_scan import gla_scan_triton
from test_gla_grad import ref_gla_scan

B, H, L, D = 2, 4, 32, 16
torch.manual_seed(42)
device = 'cuda'
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

grad_out = torch.randn_like(y_triton)
y_triton.backward(grad_out)
y_ref.backward(grad_out)

print("Forward rel err:", ((y_triton - y_ref).abs() / (y_ref.abs() + 1e-5)).max().item())
print("dq rel err:     ", ((q.grad - q_ref.grad).abs() / (q_ref.grad.abs() + 1e-5)).max().item())
print("dk rel err:     ", ((k.grad - k_ref.grad).abs() / (k_ref.grad.abs() + 1e-5)).max().item())
print("dv rel err:     ", ((v.grad - v_ref.grad).abs() / (v_ref.grad.abs() + 1e-5)).max().item())
print("dlam rel err:   ", ((lambdas.grad - lambdas_ref.grad).abs() / (lambdas_ref.grad.abs() + 1e-5)).max().item())
print("dq abs err:     ", (q.grad - q_ref.grad).abs().max().item())
print("dk abs err:     ", (k.grad - k_ref.grad).abs().max().item())
print("dv abs err:     ", (v.grad - v_ref.grad).abs().max().item())
print("dlam abs err:   ", (lambdas.grad - lambdas_ref.grad).abs().max().item())
