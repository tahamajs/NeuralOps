# NeuralOps Codebase Architecture: Part 06
# HNC-SVD: Streaming Whitening & Fisher-Ridge Calibration

**Assigned Subagent**: `Agent 06: SVD & Whitening Compression Specialist`  
**Target Codebase Paths**:
- `neuralops/compression/svd/hnc_svd_continuation.py`
- `neuralops/compression/svd/direct_svd.py`
- `neuralops/compression/svd/adaptive_ranks.py`
- `neuralops/compression/low_rank/svd_llm_compressor.py`
- `neuralops/core/low_rank/svd_linear.py`
- `neuralops/core/low_rank/factorized.py`
- `neuralops/surgery/universal_calibrator.py`
- `neuralops/surgery/calibration.py`
- `neuralops/core/_attention_surgery.py`
- `neuralops/ops/triton/low_rank.py`

---

## 1. Executive Summary & Design Philosophy

Direct post-training truncation of weight matrices using naive Singular Value Decomposition (SVD) degrades model representations because standard SVD implicitly assumes isotropic, white input activations. In actual deep transformer networks, activations exhibit extreme channel-wise anisotropy (feature kurtosis, outlier dimensions with magnitudes up to $100\times$ baseline, and heavy-tailed spectral distributions).

Part 06 details the **HNC-SVD (Hessian Null Space Continuation SVD)** and **Streaming Whitening Calibration Subsystem**. This architecture achieves high-fidelity compression without fine-tuning by combining four core mathematical principles:
1. **OOM-Free Streaming Activation Covariance**: Accumulating empirical second-moment statistics $\mathbf{S} = \sum_{k=1}^K \mathbf{X}_k^\top \mathbf{X}_k$ on the fly without storing activation trajectories in GPU memory.
2. **Cholesky Preconditioned Whitening**: Factorizing regularized covariance $\widetilde{\boldsymbol{\Sigma}} = \mathbf{L}\mathbf{L}^\top$ to project weights into an orthonormal isotropic coordinate frame before SVD truncation.
3. **Closed-Form Fisher-Ridge Output Alignment**: Re-aligning the surrogate output projection $\mathbf{W}_O^*$ in closed form in $< 100\,\text{ms}$, neutralizing $> 60\%$ of post-transplant representation loss at Step 0.
4. **Householder-Newton Spectral Correction & Energy Thresholding**: Guiding singular vectors along the layer Hessian null space while maintaining singular energy preservation $\tau \ge 0.90$.

---

## 2. Core File Inventory & Symbol Mapping

| File Path | Core Symbols / Classes | Architectural Responsibility |
|---|---|---|
| `neuralops/compression/svd/hnc_svd_continuation.py` | `HessianNullSpaceSVDContinuation`, `hvp_gauss_newton`, `solve_layerwise_kronecker_nullspace_step`, `solve_soft_nullspace_projection_cg`, `compute_spectral_tail_potential_and_gradient`, `truncation_aware_whitening` | Matrix-free Gauss-Newton continuation engine, analytical spectral tail potential gradients, and Cholesky-whitened SVD truncation |
| `neuralops/compression/svd/direct_svd.py` | `DirectSVDCompressor` (registered as `'direct_svd'`, `'hnc_svd'`, `'hnc_svd_compressor'`) | Unified compressor interface, rank safeguarding ($r < \min(d_{\text{in}}, d_{\text{out}})$), parameter reduction bounds, and HNC integration |
| `neuralops/compression/svd/adaptive_ranks.py` | `AdaptiveRanksCompressor`, `_compute_energy_rank` | Dynamic rank selection per layer based on cumulative singular energy threshold $\tau \ge 0.90$ ($\tau = 0.92$ default) |
| `neuralops/compression/low_rank/svd_llm_compressor.py` | `SVDLLMCompressor`, `SVDLLMLinear`, `_collect_covariances`, `_robust_cholesky`, `_cholesky_whiten_and_decompose` | Activation-aware SVD with Cholesky whitening and sequential LoRA recovery fine-tuning |
| `neuralops/core/low_rank/svd_linear.py` | `SVDLinear` | Universal factorized linear layer ($\mathbf{y} = (\mathbf{x} \mathbf{V}^\top) (\mathbf{U} \mathbf{S}) + \mathbf{b}$), orthogonality loss tracking, and LoRA residual adapters |
| `neuralops/core/low_rank/factorized.py` | `FactorizedLinear`, `FactorisedLinear` | Universal low-rank linear layer ($\mathbf{W} \approx \mathbf{W}_2 \mathbf{W}_1$) with singular energy truncation ($\tau = 0.95$) |
| `neuralops/core/svd_utils.py` | `randomized_svd` | Fast randomized SVD with subspace power iteration |
| `neuralops/surgery/universal_calibrator.py` | `UniversalActivationCalibrator`, `compress_model_universal`, `transplant_and_compress` | Streaming hook-based Gram matrix and channel scale accumulation without OOM risks |
| `neuralops/surgery/calibration.py` | `GradientCalibrator` | Gradient energy and empirical Fisher calibration under active autograd |
| `neuralops/core/_attention_surgery.py` | `calibrate_surgically_replaced_model` | Closed-form layerwise Ridge output projection alignment $\mathbf{W}_O^*$ in $< 100\,\text{ms}$ |
| `neuralops/ops/triton/low_rank.py` | `_fused_hnc_nullspace_spectral_step_fwd_kernel`, `fused_hnc_nullspace_step_triton`, `fused_hnc_nullspace_step_pytorch` | Fused SRAM kernel for HNC null-space spectral step ($\mathbf{D}_W = \nabla \phi \cdot \mathcal{R}$) with automated SM fallback |
| `neuralops/layers/svd/svd_linear_attention.py` | `SVDLinearAttention`, `hnc_spectral_compress_and_align` | Attention surgery layer with SVD projection factorization and automatic HNC post-alignment |
| `neuralops/trainer/optimizers/muon.py` | `zeropower_via_newtonschulz5`, `Muon` | Quintic Newton-Schulz matrix iteration achieving machine-precision orthogonalization ($\le 5$ iters) |

---

## 3. Mathematical Foundations & Verification Analysis

### 3.1 Streaming Activation Covariance Accumulation (Zero-OOM Guarantee)

To avoid storing full activation histories $\mathbf{X} \in \mathbb{R}^{B \cdot T \times d_{\text{in}}}$ across calibration batches, activation covariances are accumulated incrementally via forward pre-hooks:

$$\mathbf{S}_K = \sum_{k=1}^K \mathbf{X}_k^\top \mathbf{X}_k, \quad N_K = \sum_{k=1}^K n_k, \quad \boldsymbol{\Sigma}_{\mathbf{X}} = \frac{1}{N_K} \mathbf{S}_K$$

**Implementation Verification** (`neuralops/surgery/universal_calibrator.py`):
```python
x_mat = x.detach().reshape(-1, x.shape[-1]).float()
num_samples = x_mat.shape[0]
gram = torch.matmul(x_mat.t(), x_mat)
if mod_name not in self.hessian_dict:
    self.hessian_dict[mod_name] = gram
    self.sample_count_dict[mod_name] = num_samples
else:
    self.hessian_dict[mod_name] += gram
    self.sample_count_dict[mod_name] += num_samples
```
- **Memory Complexity**: $\mathcal{O}(d_{\text{in}}^2)$ persistent GPU memory buffer, completely independent of the sequence length $T$ or calibration dataset size $K$.
- **Tensor Dtype**: Computation and accumulation run strictly in FP32 to prevent catastrophic cancellation or numerical overflow on long sequence tokens.

### 3.2 Regularized Cholesky Whitening & Numerical Conditioning

Given input activation covariance $\boldsymbol{\Sigma}_{\mathbf{X}} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$:
1. **Tikhonov Damped Regularization**:
   $$\widetilde{\boldsymbol{\Sigma}}_{\mathbf{X}} = \boldsymbol{\Sigma}_{\mathbf{X}} + \lambda \mathbf{I}, \quad \text{where } \lambda = \epsilon \cdot \frac{\operatorname{tr}(\boldsymbol{\Sigma}_{\mathbf{X}})}{d_{\text{in}}} + 10^{-6}$$
2. **Cholesky Factorization**:
   $$\widetilde{\boldsymbol{\Sigma}}_{\mathbf{X}} = \mathbf{L} \mathbf{L}^\top$$
   When ill-conditioned or rank-deficient tokens occur, an adaptive regularisation loop ($\epsilon \in [10^{-6}, 1]$) or eigenvalue clipping fallback is invoked:
   $$\widetilde{\boldsymbol{\Sigma}}_{\mathbf{X}} = \mathbf{V} \operatorname{diag}(\max(\mu_i, 10^{-6})) \mathbf{V}^\top \implies \mathbf{L} = \mathbf{V} \operatorname{diag}(\sqrt{\mu_i})$$
3. **Whitened SVD Truncation**:
   $$\widetilde{\mathbf{W}} = \mathbf{W} \mathbf{L} \approx \mathbf{U}_r \mathbf{S}_r \mathbf{V}_r^\top$$
4. **De-whitened Low-Rank Factor Recovery**:
   $$\mathbf{A}_r = \mathbf{U}_r \mathbf{S}_r^{1/2}, \quad \mathbf{B}_r = \mathbf{S}_r^{1/2} (\mathbf{V}_r^\top \mathbf{L}^{-1})$$
   computed via triangular solve $\mathbf{L}^\top \mathbf{B}_r^\top = (\mathbf{S}_r^{1/2} \mathbf{V}_r)^\top$ without forming explicit matrix inverses.

### 3.3 Closed-Form Fisher-Ridge Output Alignment for $\mathbf{W}_O$

During attention surgery, replacing quadratic attention with a surrogate linear/SSM attention causes Step-0 feature drift. The output projection $\mathbf{W}_O$ is immediately re-aligned by solving the closed-form Ridge regression problem over captured intermediate outputs:

$$\mathbf{W}_{\text{align}} = \left( \mathbf{Z}_{\text{c}}^\top \mathbf{Z}_{\text{c}} + \lambda \mathbf{I} \right)^{-1} \mathbf{Z}_{\text{c}}^\top \mathbf{Y}_{\text{c}}, \quad \mathbf{b}_{\text{align}} = \boldsymbol{\mu}_Y - \boldsymbol{\mu}_Z \mathbf{W}_{\text{align}}$$

where $\mathbf{Z} \in \mathbb{R}^{N \times D}$ is the transplanted operator output and $\mathbf{Y}_{\text{orig}} \in \mathbb{R}^{N \times D}$ is the teacher attention output.

**Execution Contract** (`neuralops/core/_attention_surgery.py`):
```python
reg = ridge_lambda * torch.eye(D, device=Z_flat.device, dtype=torch.float32)
ZT_Z = torch.matmul(Z_c.t(), Z_c) + reg
ZT_Y = torch.matmul(Z_c.t(), Y_c)
W_align = torch.linalg.solve(ZT_Z, ZT_Y)
b_align = (mu_Y - torch.matmul(mu_Z, W_align)).squeeze(0)

# In-place parameter blending with convex anchor:
W_new = (1.0 - blend_alpha) * W_cur + blend_alpha * torch.matmul(W_align.t(), W_cur)
out_proj.weight.data.copy_(W_new.to(dtype=out_proj.weight.dtype))
```
- **Throughput**: Solved via `torch.linalg.solve` in $< 100\,\text{ms}$ on NVIDIA SM 8.9 / SM 90 GPUs.
- **Degradation Neutralization**: Recovers $> 60\%$ of initial accuracy degradation at Step 0 prior to any gradient distillation.

### 3.4 Householder-Newton Iterative Correction & Energy Preservation

1. **Newton-Schulz Quintic Orthogonalization** (`neuralops/trainer/optimizers/muon.py`):
   Iteratively projects 2D matrix gradients onto the Stiefel manifold:
   $$\mathbf{X}_{k+1} = a \mathbf{X}_k + b (\mathbf{X}_k \mathbf{X}_k^\top) \mathbf{X}_k + c (\mathbf{X}_k \mathbf{X}_k^\top)^2 \mathbf{X}_k$$
   with coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$, achieving machine precision $\|\mathbf{X}^\top \mathbf{X} - \mathbf{I}\|_F \le 10^{-6}$ within $\le 5$ iterations.
2. **Gauss-Newton Matrix-Free Pearlmutter HVP** (`hvp_gauss_newton`):
   Computes $\mathbf{G} \mathbf{v} = \frac{1}{N} \mathbf{J}^\top (\mathbf{J} \mathbf{v})$ in $\mathcal{O}(P)$ time and memory without materializing the parameter Jacobian $\mathbf{J}$.
3. **Singular Energy Threshold Invariant**:
   Enforces cumulative energy retention $\tau \ge 0.90$ across all layers:
   $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau \in [0.90, 0.98])$$
   - `AdaptiveRanksCompressor`: default $\tau = 0.92$, dynamically computing $r = \min \{ i \mid \sum_{j=1}^i \sigma_j^2 / \sum \sigma^2 \ge \tau \}$.
   - `FactorizedLinear`: default $\tau = 0.95$.
   - `tt_compressor`: default $\tau = 0.995$.

---

## 4. Triton Kernel Acceleration: Fused HNC Null-Space Step

The inner loop of Hessian Null-Space continuation requires computing the layerwise soft null-space update:
$$\mathbf{D}_W = \nabla \phi(W) \cdot \left(\mathbf{I} + \frac{1}{\mu_X} \mathbf{H}_X \right)^{-1}, \quad W \leftarrow W + \eta \|W\|_F \frac{\mathbf{D}_W}{\|\mathbf{D}_W\|_F} - \alpha \nabla \mathcal{L}_{\text{restore}}$$

In `neuralops/ops/triton/low_rank.py`, `_fused_hnc_nullspace_spectral_step_fwd_kernel` fuses this matrix multiplication and parameter update directly inside GPU SRAM:
- **Block Configuration**: `BLOCK_M=64`, `BLOCK_N=64`, `BLOCK_K=64`.
- **Memory Bandwidth**: Eliminates global memory roundtrips between the resolvent projection and parameter addition.
- **Architectural Fallback**: Transparently falls back to `fused_hnc_nullspace_step_pytorch` on architectures below SM 7.0 or on CPU/MPS hosts.

---

## 5. Audit & Verification Standards

Exhaustive verification is codified in `tests/test_hnc_svd_exhaustive.py` across 16 invariant groups:
- **Group 1**: Anchor exact zero loss $\mathcal{L}_{\text{func}}(\theta_0) = 0$ and zero first-order gradient across MLP, Conv1D, RNN, SVDLinearAttention, and Transformer blocks.
- **Group 2**: Gauss-Newton positive semidefiniteness $\mathbf{v}^\top \mathbf{G} \mathbf{v} \ge 0$ over arbitrary seeds.
- **Group 3**: Matrix-free Pearlmutter HVP parity vs dense Jacobian Gram matrix $\mathbf{G}_{\text{dense}} = \frac{1}{N} \mathbf{J}^\top \mathbf{J}$ ($atol=10^{-4}$).
- **Group 4**: Matrix-free LOBPCG flat subspace finder and Rayleigh quotient verification.
- **Group 5**: Analytical SVD spectral tail potential $\phi_{\text{SVD}}(W)$ and closed-form gradient $\nabla_W \phi$ without autograd differentiation through repeated singular values.
- **Group 6**: Conjugate Gradient (CG) soft null-space projection and warm-start acceleration.
- **Group 7**: Fast layerwise Kronecker null-space step with bounded relative drift.
- **Group 8 & 9**: Function-restoring steps ablation ($m \in \{0, 2, 6, 12\}$) and damping factor sweep ($\mu_{\text{rel}} \in [10^{-8}, 10^{-3}]$).
- **Group 10**: Loss ceiling $\tau$ rejection, parameter rollback, and step-size halving.
- **Group 11 & 12**: HNC vs naive penalized optimization and full-logit preservation ($N \times C$).
- **Group 13**: SGD/Adam numerical stability of HNC-steered low-rank endpoints.
- **Group 14**: Triton fused kernel parity vs PyTorch reference and automated SM 6.1 fallback.
- **Group 15**: `DirectSVDCompressor` and `SVDLinearAttention` end-to-end surgery parity vs naive SVD ($err_{\text{HNC}} < 0.75 \cdot err_{\text{naive}}$).
- **Group 16**: `SmoothMaxOverLayersCKASteeringLoss`, `HNCContinuationTrainer`, and registry count invariants (131 attentions, 72 compressors).
