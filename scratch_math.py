import torch

torch.manual_seed(42)
B, H, L, D = 1, 1, 4, 3
q = torch.randn(B, H, L, D, requires_grad=True)
k = torch.randn(B, H, L, D, requires_grad=True)
v = torch.randn(B, H, L, D, requires_grad=True)
lam_raw = torch.rand(B, H, L, requires_grad=True)
lam = lam_raw.clamp(min=1e-4, max=1.0 - 1e-4)

S = torch.zeros(D, D)
Z = torch.zeros(D)
S_all, Z_all = [], []
ys = []
for t in range(L):
    S = lam[0, 0, t] * S + torch.outer(k[0, 0, t], v[0, 0, t])
    Z = lam[0, 0, t] * Z + k[0, 0, t]
    S_all.append(S)
    Z_all.append(Z)
    num = torch.matmul(q[0, 0, t:t+1], S)
    den = (q[0, 0, t] * Z).sum().clamp_min(1e-4)
    ys.append(num / den)

y = torch.cat(ys, dim=0)
grad_out = torch.randn_like(y)
y.backward(grad_out)

print("--- AUTOGRAD GRADS ---")
print("q.grad:\n", q.grad[0, 0])
print("k.grad:\n", k.grad[0, 0])
print("v.grad:\n", v.grad[0, 0])
print("lam.grad:\n", lam_raw.grad[0, 0])

# Analytical backward in Python:
dS = torch.zeros(D, D)
dZ = torch.zeros(D)
dq_ana = torch.zeros(L, D)
dk_ana = torch.zeros(L, D)
dv_ana = torch.zeros(L, D)
dlam_ana = torch.zeros(L)

for t in reversed(range(L)):
    qt = q[0, 0, t]
    kt = k[0, 0, t]
    vt = v[0, 0, t]
    lam_t = lam[0, 0, t]
    dy_t = grad_out[t]
    St = S_all[t]
    Zt = Z_all[t]

    num_t = torch.matmul(qt.unsqueeze(0), St).squeeze(0)
    den_raw = (qt * Zt).sum()
    den_t = den_raw.clamp_min(1e-4)

    dnum_t = dy_t / den_t
    dden_t = - (dy_t * num_t).sum() / (den_t * den_t) if den_raw > 1e-4 else 0.0

    dS = dS + torch.outer(qt, dnum_t)
    dZ = dZ + qt * dden_t

    dq_ana[t] = torch.matmul(St, dnum_t) + Zt * dden_t
    dv_ana[t] = torch.matmul(kt, dS) # wait: sum_i kt[i] * dS[i, j] = kt @ dS
    dk_ana[t] = torch.matmul(dS, vt) + dZ # sum_j dS[i, j] * vt[j] = dS @ vt

    if t > 0:
        S_prev = S_all[t - 1]
        Z_prev = Z_all[t - 1]
        dlam_ana[t] = (dS * S_prev).sum() + (dZ * Z_prev).sum()
    else:
        dlam_ana[t] = 0.0

    dS = lam_t * dS
    dZ = lam_t * dZ

print("\n--- ANALYTICAL GRADS ---")
print("dq_ana:\n", dq_ana)
print("dk_ana:\n", dk_ana)
print("dv_ana:\n", dv_ana)
print("dlam_ana:\n", dlam_ana)

print("\n--- DIFFS ---")
print("dq max diff:", (q.grad[0, 0] - dq_ana).abs().max().item())
print("dk max diff:", (k.grad[0, 0] - dk_ana).abs().max().item())
print("dv max diff:", (v.grad[0, 0] - dv_ana).abs().max().item())
print("dlam max diff:", (lam_raw.grad[0, 0] - dlam_ana).abs().max().item())
