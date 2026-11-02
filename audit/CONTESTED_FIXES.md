# Stage 3 Contested Fixes & Root-Cause Forensic Evaluation

Detailed technical diagnosis and audit of the 11 contested kernel cases from `audit/CODE_FIX_PROPOSALS_2026-11-02.md`.

---

## Group A — Numerical Mismatch (Boundary Handling vs. Clamping)

### 1. `rfa_attention` (WP2.1)
- **Report Proposed Fix:** Clamp intermediate state `s` and `denom` to $\pm 65504$ (FP16 dynamic range maximum) and clamp `norm` to `1e-4 * sign(denom)`.
- **Reference Analysis:**
  - `neuralops/ops/cuda/rfa_attention/rfa_attention_autograd.py:31-41` computes in full FP32 precision:
    `s = s + phi_k_f[:, :, t, :] * v_f[:, :, t, :]`
    `denom = denom + phi_k_f[:, :, t, :]`
    `norm = torch.clamp(torch.abs(denom), min=1e-5)`
    `out[:, :, t, :] = ((phi_q_f[:, :, t, :] * s) / norm).to(phi_q.dtype)`
  - The reference does NOT clamp `s` or `denom` to 65504.
- **Root Cause & Evaluation:**
  - Clamping `s` and `denom` in the CUDA kernel introduces divergence from the PyTorch reference implementation.
  - The CUDA kernel (`neuralops/ops/cuda/rfa_attention/rfa_attention.cu:37-49`) already computes `s`, `denom`, and `norm` in FP32 with identical formula `norm = fmaxf(fabsf(denom), 1e-5f)`.
  - Non-finite guard `isnan(res) || isinf(res) ? 0.0f : res` already protects against invalid output values without truncating valid FP32 dynamic ranges.
- **Decision:** **REJECT REPORT FIX**. Keep FP32 precision aligned with the PyTorch reference; do not introduce arbitrary $\pm 65504$ clamps that diverge from reference.

---

### 2. `phantom` (WP2.4)
- **Report Proposed Fix:** Replace standard floating-point functions with fast-math intrinsics: `__fdividef(1.0f, 1.0f + __expf(-raw_gate))`.
- **Reference Analysis:**
  - Reference `neuralops/ops/cuda/phantom/phantom_autograd.py:44`:
    `g = torch.sigmoid(gate.float())`
    `mask = torch.where(g >= threshold, g, torch.zeros_like(g)).to(x.dtype)`
  - PyTorch's `torch.sigmoid` uses IEEE-754 compliant rounding.
- **Root Cause & Evaluation:**
  - Using `__fdividef` and `__expf` fast-math instructions lowers numerical precision to 2-3 ULP (Units in the Last Place), *increasing* divergence against `torch.sigmoid`.
  - The numerical divergence in test suites occurs at the discontinuous Heaviside step boundary where `g ≈ threshold`: slight rounding differences between CPU IEEE-754 and CUDA FP32 place values on opposite sides of `g >= threshold`, resulting in difference of $|g \cdot x| \approx 0.15$.
- **Decision:** **REJECT REPORT FIX**. Fast-math makes accuracy worse. Boundary discrepancy is an inherent property of step function quantization across hardware backends; parity checking must allow boundary tolerance (`PASS_HIGH_DIFF`).

---

### 3. `spectra_v2` (WP2.5)
- **Report Proposed Fix:** Shift the threshold comparison boundary by an arbitrary epsilon:
  `const float filter = (fabsf(freq) >= (threshold - 1e-6f)) ? 1.0f : 0.0f;`
- **Reference Analysis:**
  - Reference `neuralops/ops/cuda/spectra_v2/spectra_v2_autograd.py:32`:
    `mask = (freq_bands.abs() > threshold).to(x.dtype)`
    `return x * mask`
  - CUDA implementation (`spectra_v2.cu:34`):
    `const float filter = (fabsf(freq) > threshold) ? 1.0f : 0.0f;`
- **Root Cause & Evaluation:**
  - Shifting threshold by $-1\times 10^{-6}$ in CUDA only moves the discontinuity boundary, immediately causing divergence on any values in `[threshold - 1e-6, threshold]` where PyTorch reference yields 0 but CUDA yields 1.
  - The CUDA formula and Python reference already have mathematically identical expressions: both evaluate `|freq| > threshold`.
- **Decision:** **REJECT REPORT FIX**. Do not apply artificial threshold shifts that violate reference alignment.

---

## Group B — Structural Kernel Fixes

### 4. `paged_recurrent_step` (WP2.3)
- **Report Proposed Fix:** Shared-memory staging for `q` and `k` vectors to reduce latency.
- **Reference Analysis:**
  - `paged_recurrent_step_autograd.py:53-56`:
    `S_curr = page_pool[phys_pages]`
    `kv = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))`
    `S_next = decay * S_curr + kv`
    `return torch.matmul(q.unsqueeze(-2), S_next).squeeze(-2)`
  - PyTorch accumulates `S_next` in FP32.
- **Root Cause & Evaluation:**
  - The report's proposed fix only addresses latency via shared memory loading of `q` and `k`, but ignores the primary source of numerical error ($0.2846$ divergence): in-place FP16 state quantization in `page_pool[s_idx] = from_float<scalar_t>(s_new)`.
  - Staging `q` and `k` without upgrading `page_pool` to FP32 leaves the numerical error unresolved.
  - Upgrading `page_pool` to FP32 across all paged attention systems requires architectural memory allocation updates in the paged KV cache manager.
- **Decision:** **REJECT PARTIAL FIX**. Per mission instructions, a partial fix that leaves either latency or numerical error broken is rejected. Retained in GPU verification queue for comprehensive memory redesign.

---

## Group C — Timeout Kernels

### 5. `rotational_svd` (WP3.1)
- **Report Diagnosis:** Warp ID return stall (`if (warp_id > 0) return;`) with 128 threads per block causing barrier deadlock in WMMA kernel.
- **Evaluation:** **VERIFIED**. Launching 128 threads with `warp_id > 0` returning immediately leaves inactive threads in the thread block, violating WMMA collective execution assumptions on modern architectures.
- **Action:** Fixed in commit `d8c69ba0fb` (`fix(cuda): rotational_svd — launch single warp (32 threads) per 16x16 WMMA block`).

---

### 6. `rpca_sparse_svd` (WP3.2)
- **Report Diagnosis:** Unhandled sparse tensor CSR/COO conversion deadlock in JIT compilation path.
- **Evaluation:** **DIAGNOSIS REFUTED / CLARIFIED**.
  - `neuralops/ops/cuda/rpca_sparse_svd/rpca_sparse_svd_autograd.py:46` loads via `load_cuda_module(..., functions=["rpca_sparse_svd_fwd"])`.
  - `rpca_sparse_svd.cu` is already dense GEMM (`at::matmul`, `at::matmul_out`, `out.addmm_`).
  - The kernel contains no unbounded loops, no synchronizations, and dense tensor operations. It is not an infinite loop.
- **Decision:** **REJECT REPORT CLAIMS OF SPARSE DEADLOCK**. Dense GEMM kernel is well-formed.

---

### 7. `tt_contraction` (WP3.3)
- **Report Diagnosis:** "Grid dimension integer overflow when contracting large multi-index tensor train cores."
- **Evaluation:** **DIAGNOSIS REFUTED**.
  - `tt_contraction.cu:90-93`:
    `const int total = N * K;`
    `const int threads = 256;`
    `const int blocks = (total + threads - 1) / threads;`
  - In standard benchmarks, `N=256, K=256 => total=65,536 => blocks=256`. Even for large hidden dimensions ($N=4096, K=4096$), `total=16,777,216 => blocks=65,536`.
  - Modern CUDA architectures support grid dimensions up to $2^{31} - 1$ in `gridDim.x`. 12,288 or 65,536 is thousands of times below the hardware limit.
- **Decision:** **REJECT REPORT DIAGNOSIS**. No grid overflow exists.

---

### 8. `samba_moe` (WP3.6)
- **Report Diagnosis:** "Total token count integer calculation overflow and grid launch boundary mismatch."
- **Evaluation:** **DIAGNOSIS REFUTED / SELF-CONTRADICTORY**.
  - Total tokens calculation: `const int total = expert_out1.numel(); const int blocks = (total + THREADS - 1) / THREADS;`
  - A 1D elementwise combination kernel has zero inter-thread communication, zero barrier synchronizations (`__syncthreads`), and straightforward grid indexing.
  - The report claims an integer overflow while stating in the next sentence that `blocks = 12,288`, which is well within 32-bit integer and CUDA grid limits.
- **Decision:** **REJECT REPORT DIAGNOSIS**.

---

### 9. `sparse_attention` (WP3.7)
- **Report Diagnosis:** "Deadlock on s_num_active gathering when block_mask contains 0 active blocks."
- **Evaluation:** **DIAGNOSIS REFUTED**.
  - In `neuralops/ops/cuda/sparse_attention/sparse_attention.cu:155-168`:
    ```cpp
    if (l_prev[qi] > 0.0f) {
        const float inv_l = 1.0f / l_prev[qi];
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            out[out_base + lane + e * 32] = acc[qi][e] * inv_l;
        }
    } else {
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            out[out_base + lane + e * 32] = 0.0f;
        }
    }
    ```
  - When `s_num_active == 0`, `l_prev[qi]` remains `0.0f`, cleanly taking the `else` branch which zeroes the output. No division by zero, no NaN, and no infinite loop occurs.
- **Decision:** **REJECT REPORT DIAGNOSIS**. Code already contains the zero-active-block guard.
