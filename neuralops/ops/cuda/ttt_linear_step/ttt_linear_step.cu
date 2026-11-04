// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__global__ void ttt_linear_step_scan_fwd_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ etas,
    scalar_t* __restrict__ out,
    const int B,
    const int H,
    const int N,
    const int D,
    const int eta_stride_h
) {
    // Each thread block processes one batch-head sequence (b, h)
    const int bh = blockIdx.x;
    if (bh >= B * H) return;

    const int b = bh / H;
    const int h = bh % H;

    const int bh_offset = bh * (N * D);
    const int eta_offset = b * (N * eta_stride_h) + h;

    // Dynamic shared memory allocation:
    // s_S: D * D floats (recurrent memory state matrix)
    // s_k: D floats
    // s_v: D floats
    // s_q: D floats
    // s_pred: D floats
    // s_red: 32 floats (warp reduction scratchpad)
    extern __shared__ float s_mem[];
    float* s_S = s_mem;
    float* s_k = s_mem + D * D;
    float* s_v = s_mem + D * D + D;
    float* s_q = s_mem + D * D + 2 * D;
    float* s_pred = s_mem + D * D + 3 * D;
    float* s_red = s_mem + D * D + 4 * D;

    // Initialize state matrix S to zeros
    const int total_s_elements = D * D;
    for (int idx = threadIdx.x; idx < total_s_elements; idx += blockDim.x) {
        s_S[idx] = 0.0f;
    }
    __syncthreads();

    // Iterate through sequence tokens t = 0 .. N - 1
    for (int t = 0; t < N; ++t) {
        const int tok_offset = bh_offset + t * D;
        const scalar_t* q_t = q + tok_offset;
        const scalar_t* k_t = k + tok_offset;
        const scalar_t* v_t = v + tok_offset;
        scalar_t* out_t = out + tok_offset;

        const float eta = to_float(etas[eta_offset + t * eta_stride_h]);

        // Load k_t, v_t, q_t into shared memory
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            s_k[d] = to_float(k_t[d]);
            s_v[d] = to_float(v_t[d]);
            s_q[d] = to_float(q_t[d]);
        }
        __syncthreads();

        // 1. Compute pred_v = k_t @ S (dimension 1 x D)
        // pred_v[j] = sum_i (k_t[i] * S[i, j])
        for (int j = threadIdx.x; j < D; j += blockDim.x) {
            float sum_j = 0.0f;
            #pragma unroll 4
            for (int i = 0; i < D; ++i) {
                sum_j = fmaf(s_k[i], s_S[i * D + j], sum_j);
            }
            // err_v[j] = pred_v[j] - v_t[j]
            s_pred[j] = sum_j - s_v[j];
        }
        __syncthreads();

        // 2. Online gradient update on state matrix:
        // S = S - eta * (k_t^T @ err_v)
        // S[i, j] = S[i, j] - eta * (k_t[i] * s_pred[j])
        for (int idx = threadIdx.x; idx < total_s_elements; idx += blockDim.x) {
            const int i = idx / D;
            const int j = idx % D;
            s_S[idx] = fmaf(-eta * s_k[i], s_pred[j], s_S[idx]);
        }
        __syncthreads();

        // 3. Compute output y_t = q_t @ S (dimension 1 x D)
        // y_t[j] = sum_i (q_t[i] * S[i, j])
        for (int j = threadIdx.x; j < D; j += blockDim.x) {
            float out_j = 0.0f;
            #pragma unroll 4
            for (int i = 0; i < D; ++i) {
                out_j = fmaf(s_q[i], s_S[i * D + j], out_j);
            }
            out_t[j] = from_float<scalar_t>(out_j);
        }
        __syncthreads();
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor ttt_linear_step_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& etas
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(etas);

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, D]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape mismatch");

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    // etas can be [B, H, N, 1] or [B, H, N]
    const int eta_stride_h = etas.stride(1);

    // Threads per block: choose 128 or 256 depending on D
    int threads = 128;
    if (D >= 64) {
        threads = 256;
    }

    // Shared memory: s_S(D*D) + s_k(D) + s_v(D) + s_q(D) + s_pred(D) + s_red(32)
    const size_t shared_mem_bytes = (D * D + 4 * D + 32) * sizeof(float);

    dim3 grid(B * H);
    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "ttt_linear_step_fwd", ([&] {
            neuralops::cuda::ttt_linear_step_scan_fwd_kernel<scalar_t>
                <<<grid, threads, shared_mem_bytes, stream>>>(
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(),
                    etas.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    B, H, N, D,
                    eta_stride_h
                );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
