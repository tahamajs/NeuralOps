/*
 * Hadamard Linear Attention (HLA) CUDA C++ Kernel
 *
 * Hardware-accelerated causal linear attention with per-head exponential decay
 * and normalizer vector.
 *
 * Author: Taha Majlesi
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t, int MAX_D>
__launch_bounds__(64)
__global__ void hla_attention_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ decay,
    const int B,
    const int H,
    const int L,
    const int D,
    const float eps,
    const int decay_stride_b,
    const int decay_stride_h,
    const int decay_stride_l
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    __shared__ float s_S[MAX_D][MAX_D];
    __shared__ float s_z[MAX_D];
    __shared__ float s_k[MAX_D];
    __shared__ float s_v[MAX_D];
    __shared__ float s_q[MAX_D];

    // Initialize recurrent state
    for (int idx = tid; idx < D * D; idx += num_threads) {
        int r = idx / D;
        int c = idx % D;
        s_S[r][c] = 0.0f;
    }
    if (tid < D) {
        s_z[tid] = 0.0f;
    }
    __syncthreads();

    const int base_bh = (b * H + h) * L;

    for (int t = 0; t < L; ++t) {
        const int offset = (base_bh + t) * D;
        const int decay_idx = b * decay_stride_b + h * decay_stride_h + t * decay_stride_l;
        const float gamma = decay[decay_idx];

        if (tid < D) {
            s_q[tid] = to_float(q[offset + tid]);
            s_k[tid] = to_float(k[offset + tid]);
            s_v[tid] = to_float(v[offset + tid]);
            s_z[tid] = gamma * s_z[tid] + s_k[tid];
        }

        // Update state S = gamma * S + k^T * v
        for (int idx = tid; idx < D * D; idx += num_threads) {
            int r = idx / D;
            int c = idx % D;
            s_S[r][c] = gamma * s_S[r][c] + s_k[r] * s_v[c];
        }
        __syncthreads();

        // Compute denominator = q @ z
        float local_denom = 0.0f;
        for (int i = tid; i < D; i += num_threads) {
            local_denom += s_q[i] * s_z[i];
        }
        local_denom = block_reduce_sum(local_denom);
        __shared__ float s_denom;
        if (tid == 0) {
            s_denom = fmaxf(local_denom, eps);
        }
        __syncthreads();
        const float denom = s_denom;

        // Compute out = (q @ S) / denom
        for (int col = tid; col < D; col += num_threads) {
            float num = 0.0f;
            for (int r = 0; r < D; ++r) {
                num += s_q[r] * s_S[r][col];
            }
            out[offset + col] = from_float<scalar_t>(num / denom);
        }
        __syncthreads();
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor hla_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& decay,
    double eps
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D = q.size(3);

    TORCH_CHECK(k.sizes() == q.sizes(), "k shape must match q");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape must match q");
    TORCH_CHECK(D <= 64, "hla_attention_fwd supports head_dim <= 64, got ", D);

    auto decay_f = decay.contiguous().to(torch::kFloat32);
    int d_sb = 0, d_sh = 0, d_sl = 0;
    if (decay_f.dim() == 4) {
        decay_f = decay_f.squeeze(-1); // (B, H, L) or (1, H, 1)
    }
    if (decay_f.dim() == 3) {
        d_sb = (decay_f.size(0) > 1) ? decay_f.stride(0) : 0;
        d_sh = (decay_f.size(1) > 1) ? decay_f.stride(1) : 0;
        d_sl = (decay_f.size(2) > 1) ? decay_f.stride(2) : 0;
    } else if (decay_f.dim() == 2) {
        d_sh = decay_f.stride(0);
        d_sl = (decay_f.size(1) > 1) ? decay_f.stride(1) : 0;
    } else if (decay_f.dim() == 1) {
        d_sh = decay_f.stride(0);
    }

    auto out = torch::empty_like(q);
    const dim3 grid(B, H);
    const int threads = 64;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "hla_attention_fwd", ([&] {
            neuralops::cuda::hla_attention_fwd_kernel<scalar_t, 64><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                decay_f.data_ptr<float>(),
                B, H, L, D, static_cast<float>(eps),
                d_sb, d_sh, d_sl
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
