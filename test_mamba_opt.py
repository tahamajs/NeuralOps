import time
import torch
import torch.nn.functional as F

B, L, D, S = 128, 128, 64, 16
x_conv = torch.randn(B, L, D, requires_grad=True)
delta = torch.rand(B, L, D, requires_grad=True) * 0.1 + 0.01
B_mat = torch.randn(B, L, S, requires_grad=True)
C_mat = torch.randn(B, L, S, requires_grad=True)
A_log = torch.randn(S, requires_grad=True)

A = -torch.exp(A_log)

# 1. Original Loop
start_t = time.time()
h = x_conv.new_zeros(B, D, S)
outputs = []
for t in range(L):
    dt = delta[:, t]                      # [B, D]
    decay = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)) # [B, D, S]
    input_term = dt.unsqueeze(-1) * B_mat[:, t, None, :] * x_conv[:, t, :, None] # [B, D, S]
    h = decay * h + input_term
    y_t = (C_mat[:, t, None, :] * h).sum(dim=-1) # [B, D]
    outputs.append(y_t)
y_loop = torch.stack(outputs, dim=1) # [B, L, D]
loss_loop = y_loop.sum()
loss_loop.backward()
loop_time = time.time() - start_t
print(f"Loop time (Fwd+Bwd): {loop_time*1000:.2f} ms")

# Fresh tensors for vec
x_conv2 = x_conv.detach().clone().requires_grad_(True)
delta2 = delta.detach().clone().requires_grad_(True)
B_mat2 = B_mat.detach().clone().requires_grad_(True)
C_mat2 = C_mat.detach().clone().requires_grad_(True)
A_log2 = A_log.detach().clone().requires_grad_(True)
A2 = -torch.exp(A_log2)

# 2. Vectorized Associative Scan / Chunked
start_t = time.time()
log_alpha = delta2.unsqueeze(-1) * A2.unsqueeze(0).unsqueeze(0).unsqueeze(0) # [B, L, D, S]
u = delta2.unsqueeze(-1) * B_mat2.unsqueeze(2) * x_conv2.unsqueeze(-1) # [B, L, D, S]

chunk_size = 16
h_running = x_conv2.new_zeros(B, D, S)
y_chunks = []
num_chunks = L // chunk_size

mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x_conv2.device, dtype=torch.bool))

for c in range(num_chunks):
    start_idx = c * chunk_size
    end_idx = start_idx + chunk_size
    
    u_c = u[:, start_idx:end_idx] # [B, K, D, S]
    log_a_c = log_alpha[:, start_idx:end_idx] # [B, K, D, S]
    lambda_c = torch.cumsum(log_a_c, dim=1) # [B, K, D, S]
    
    # lam: [B, D, S, K]
    lam = lambda_c.permute(0, 2, 3, 1)
    lam_diff = lam.unsqueeze(-1) - lam.unsqueeze(-2) # [B, D, S, K, K]
    lam_diff = torch.where(mask, lam_diff, torch.tensor(-1e4, device=x_conv2.device))
    decay_weights = torch.exp(lam_diff) # [B, D, S, K, K]
    
    h_from_prev = torch.exp(lambda_c) * h_running.unsqueeze(1) # [B, K, D, S]
    
    u_perm = u_c.permute(0, 2, 3, 1) # [B, D, S, K]
    h_intra = torch.matmul(decay_weights, u_perm.unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2) # [B, K, D, S]
    
    h_chunk = h_from_prev + h_intra # [B, K, D, S]
    y_c = (C_mat2[:, start_idx:end_idx].unsqueeze(2) * h_chunk).sum(dim=-1) # [B, K, D]
    y_chunks.append(y_c)
    
    h_running = h_chunk[:, -1]

y_vec = torch.cat(y_chunks, dim=1)
loss_vec = y_vec.sum()
loss_vec.backward()
vec_time = time.time() - start_t
print(f"Chunked Vec time (Fwd+Bwd): {vec_time*1000:.2f} ms")
diff = (y_loop - y_vec).abs().max().item()
print(f"Max numerical difference in output: {diff:.2e}")
grad_diff = (x_conv.grad - x_conv2.grad).abs().max().item()
print(f"Max numerical difference in grads: {grad_diff:.2e}")
print(f"Speedup: {loop_time / vec_time:.2f}x")
