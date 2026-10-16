#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Chunked Delta-Gate Recurrent Scan Kernel
// q: [B, H, L, D_K], k: [B, H, L, D_K], v: [B, H, L, D_V]
// alpha: [B, H, L], beta: [B, H, L]
// out: [B, H, L, D_V]
// Recurrence: S = alpha_t * S; pred_v = S @ k_t; v_corr = v_t - pred_v; S += beta_t * outer(v_corr, k_t); y_t = S @ q_t
template <typename scalar_t, int MAX_DK, int MAX_DV>
__global__ void chunked_delta_gate_scan_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ alpha,
    const scalar_t* __restrict__ beta,
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

    // Allocate shared memory state matrix S of shape [MAX_DV, MAX_DK]
    __shared__ float s_state[MAX_DV][MAX_DK];

    // Initialize state to zero
    for (int idx = tid; idx < D_V * D_K; idx += num_threads) {
        int r = idx / D_K;
        int c = idx % D_K;
        s_state[r][c] = 0.0f;
    }
    __syncthreads();

    // Loop over sequence length L
    for (int t = 0; t < L; ++t) {
        const int bh_offset = (b * H + h) * L + t;
        const float a_t = to_float(alpha[bh_offset]);
        const float b_t = to_float(beta[bh_offset]);

        const int qk_offset = ((b * H + h) * L + t) * D_K;
        const int v_offset = ((b * H + h) * L + t) * D_V;

        // Step 1: scale S by alpha_t
        for (int idx = tid; idx < D_V * D_K; idx += num_threads) {
            int r = idx / D_K;
            int c = idx % D_K;
            s_state[r][c] *= a_t;
        }
        __syncthreads();

        // Step 2: compute pred_v = S @ k_t for each row r in D_V
        for (int r = tid; r < D_V; r += num_threads) {
            float pred = 0.0f;
            for (int c = 0; c < D_K; ++c) {
                pred += s_state[r][c] * to_float(k[qk_offset + c]);
            }
            float v_val = to_float(v[v_offset + r]);
            float v_corr = v_val - pred;

            // Step 3: update row r of S: S[r, c] += b_t * v_corr * k_t[c]
            for (int c = 0; c < D_K; ++c) {
                s_state[r][c] += b_t * v_corr * to_float(k[qk_offset + c]);
            }
        }
        __syncthreads();

        // Step 4: compute out_t = S @ q_t for each row r in D_V
        for (int r = tid; r < D_V; r += num_threads) {
            float y_val = 0.0f;
            for (int c = 0; c < D_K; ++c) {
                y_val += s_state[r][c] * to_float(q[qk_offset + c]);
            }
            out[v_offset + r] = from_float<scalar_t>(y_val);
        }
        __syncthreads();
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor chunked_delta_gate_scan_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& alpha,
    const torch::Tensor& beta
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(alpha);
    CHECK_INPUT(beta);

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
        q.scalar_type(), "chunked_delta_gate_scan_fwd", ([&] {
            neuralops::cuda::chunked_delta_gate_scan_fwd_kernel<scalar_t, 64, 64><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(), alpha.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(),
                B, H, L, D_K, D_V
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
