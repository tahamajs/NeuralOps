import time
import torch

B, L, D, S = 128, 128, 64, 16
x_conv = torch.randn(B, L, D, requires_grad=True)
delta = torch.rand(B, L, D, requires_grad=True) * 0.1 + 0.01
B_mat = torch.randn(B, L, S, requires_grad=True)
C_mat = torch.randn(B, L, S, requires_grad=True)
A_log = torch.randn(S, requires_grad=True)
A = -torch.exp(A_log)

# JIT-compiled recurrence
@torch.jit.script
def ssm_recurrence_jit(x_conv: torch.Tensor, delta: torch.Tensor, B_mat: torch.Tensor, C_mat: torch.Tensor, A: torch.Tensor):
    B = x_conv.size(0)
    L = x_conv.size(1)
    D = x_conv.size(2)
    S = A.size(0)
    
    h = torch.zeros((B, D, S), dtype=x_conv.dtype, device=x_conv.device)
    outputs = []
    
    for t in range(L):
        dt = delta[:, t] # [B, D]
        decay = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)) # [B, D, S]
        input_term = dt.unsqueeze(-1) * B_mat[:, t, None, :] * x_conv[:, t, :, None] # [B, D, S]
        h = decay * h + input_term
        y_t = (C_mat[:, t, None, :] * h).sum(dim=-1) # [B, D]
        outputs.append(y_t)
        
    return torch.stack(outputs, dim=1)

# Warmup JIT
_ = ssm_recurrence_jit(x_conv, delta, B_mat, C_mat, A)

start_t = time.time()
y_jit = ssm_recurrence_jit(x_conv, delta, B_mat, C_mat, A)
loss = y_jit.sum()
loss.backward()
jit_time = time.time() - start_t
print(f"JIT time (Fwd+Bwd): {jit_time*1000:.2f} ms")
