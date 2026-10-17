#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Causal Linear Attention Recurrent Scan Kernel
// q, k: [B, H, L, D_K], v: [B, H, L, D_V] -> out: [B, H, L, D_V]
// Recurrence: S_t = S_{t-1} + k_t^T @ v_t (matrix [D_K, D_V])
//             z_t = z_{t-1} + k_t       (vector [D_K])
//             out_t = (q_t @ S_t) / max(q_t @ z_t, 1e-5)
template <typename scalar_t, int MAX_DK, int MAX_DV>
__launch_bounds__(64)
__global__ void linear_attention_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int L,
    const int D_K,
    const int D_V
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Shared memory for S [MAX_DK, MAX_DV] and z [MAX_DK]
    __shared__ float s_S[MAX_DK][MAX_DV];
    __shared__ float s_z[MAX_DK];

    // Initialize state
    for (int idx = tid; idx < D_K * D_V; idx += num_threads) {
        int r = idx / D_V;
        int c = idx % D_V;
        s_S[r][c] = 0.0f;
    }
    if (tid < D_K) {
        s_z[tid] = 0.0f;
    }
    __syncthreads();

    const int base_bh = (b * H + h) * L;

    for (int t = 0; t < L; ++t) {
        const int qk_offset = (base_bh + t) * D_K;
        const int v_offset = (base_bh + t) * D_V;

        // Step 1: Update z += k_t
        if (tid < D_K) {
            s_z[tid] += to_float(k[qk_offset + tid]);
        }

        // Step 2: Update S += k_t^T @ v_t
        for (int idx = tid; idx < D_K * D_V; idx += num_threads) {
            int r = idx / D_V; // dim in D_K
            int c = idx % D_V; // dim in D_V
            s_S[r][c] += to_float(k[qk_offset + r]) * to_float(v[v_offset + c]);
        }
        __syncthreads();

        // Step 3: Compute denominator = q_t @ z
        float local_denom = 0.0f;
        for (int i = tid; i < D_K; i += num_threads) {
            local_denom += to_float(q[qk_offset + i]) * s_z[i];
        }
        local_denom = block_reduce_sum(local_denom);
        __shared__ float s_denom;
        if (tid == 0) {
            s_denom = fmaxf(local_denom, 1e-5f);
        }
        __syncthreads();
        const float denom = s_denom;

        // Step 4: Compute out_t = (q_t @ S) / denom
        for (int col = tid; col < D_V; col += num_threads) {
            float num = 0.0f;
            for (int r = 0; r < D_K; ++r) {
                num += to_float(q[qk_offset + r]) * s_S[r][col];
            }
            out[v_offset + col] = from_float<scalar_t>(num / denom);
        }
        __syncthreads();
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor linear_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D_K = q.size(3);
    const int D_V = v.size(3);

    auto out = torch::empty_like(v);
    const dim3 grid(B, H);
    const int threads = 64;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "linear_attention_fwd", ([&] {
            neuralops::cuda::linear_attention_fwd_kernel<scalar_t, 64, 64><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(), B, H, L, D_K, D_V
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
