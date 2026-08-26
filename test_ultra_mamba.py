import time
import torch

B, L, D, S = 32, 128, 768, 16
x_conv = torch.randn(B, L, D, requires_grad=True)
delta = torch.rand(B, L, D, requires_grad=True) * 0.1 + 0.01
B_mat = torch.randn(B, L, S, requires_grad=True)
C_mat = torch.randn(B, L, S, requires_grad=True)
A_log = torch.randn(S, requires_grad=True)
A = -torch.exp(A_log)

# 1. Exact Reference Loop
h = x_conv.new_zeros(B, D, S)
outputs = []
for t in range(L):
    dt = delta[:, t]
    decay = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    input_term = dt.unsqueeze(-1) * B_mat[:, t, None, :] * x_conv[:, t, :, None]
    h = decay * h + input_term
    y_t = (C_mat[:, t, None, :] * h).sum(dim=-1)
    outputs.append(y_t)
y_ref = torch.stack(outputs, dim=1)
loss_ref = y_ref.sum()
loss_ref.backward()

# 2. Ultra-Light Memory-Efficient Chunked Scan
x_conv2 = x_conv.detach().clone().requires_grad_(True)
delta2 = delta.detach().clone().requires_grad_(True)
B_mat2 = B_mat.detach().clone().requires_grad_(True)
C_mat2 = C_mat.detach().clone().requires_grad_(True)
A_log2 = A_log.detach().clone().requires_grad_(True)
A2 = -torch.exp(A_log2)

start_t = time.time()
log_alpha = delta2.unsqueeze(-1) * A2.view(1, 1, 1, -1) # [B, L, D, S]
u = delta2.unsqueeze(-1) * B_mat2.unsqueeze(2) * x_conv2.unsqueeze(-1) # [B, L, D, S]

chunk_size = 16
num_chunks = L // chunk_size
h_running = x_conv2.new_zeros(B, D, S)
y_chunks = []

for c in range(num_chunks):
    s_idx = c * chunk_size
    e_idx = s_idx + chunk_size
    
    u_c = u[:, s_idx:e_idx] # [B, K, D, S]
    log_a_c = log_alpha[:, s_idx:e_idx] # [B, K, D, S]
    lambda_c = torch.cumsum(log_a_c, dim=1) # [B, K, D, S]
    
    decay_c = torch.exp(lambda_c) # in (0, 1]
    decay_inv_c = torch.exp(-lambda_c) # bounded within chunk!
    
    # Intra-chunk state: decay_c * cumsum(decay_inv_c * u_c)
    u_scaled = decay_inv_c * u_c
    h_intra = decay_c * torch.cumsum(u_scaled, dim=1)
    
    # Inter-chunk state: decay_c * h_running
    h_inter = decay_c * h_running.unsqueeze(1)
    
    h_chunk = h_intra + h_inter
    y_c = (C_mat2[:, s_idx:e_idx].unsqueeze(2) * h_chunk).sum(dim=-1)
    y_chunks.append(y_c)
    
    h_running = h_chunk[:, -1]

y_fast = torch.cat(y_chunks, dim=1)
loss_fast = y_fast.sum()
loss_fast.backward()
fast_time = time.time() - start_t

print(f"Fast Chunked Scan Fwd+Bwd: {fast_time*1000:.2f} ms")
diff = (y_ref - y_fast).abs().max().item()
print(f"Max difference in output: {diff:.2e}")
grad_diff = (x_conv.grad - x_conv2.grad).abs().max().item()
print(f"Max difference in grads: {grad_diff:.2e}")
