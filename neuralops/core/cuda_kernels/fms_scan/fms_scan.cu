#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Factorised Matrix State (FMS) Recurrent Scan Kernel
template <typename scalar_t, int MAX_D>
__launch_bounds__(128)
__global__ void fms_scan_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const float* __restrict__ decay,
    const int B,
    const int H,
    const int L,
    const int D,
    const float inv_sqrt_d
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int h = bh % H;
    const float d_val = decay[h];
    const int j = threadIdx.x;

    // Each thread j holds column j of state S
    float S_col[MAX_D];
    #pragma unroll
    for (int i = 0; i < MAX_D; ++i) {
        S_col[i] = 0.0f;
    }

    __shared__ float s_k[MAX_D];
    __shared__ float s_q[MAX_D];
    __shared__ float s_z[MAX_D];
    __shared__ float s_inv_den;

    if (j == 0) {
        #pragma unroll
        for (int i = 0; i < MAX_D; ++i) {
            s_z[i] = 0.0f;
        }
    }
    __syncthreads();

    const int head_offset = bh * (L * D);

    for (int t = 0; t < L; ++t) {
        const int tok_offset = head_offset + t * D;

        float v_j = 0.0f;
        if (j < D && j < MAX_D) {
            const float q_val = to_float(Q[tok_offset + j]);
            const float k_val = to_float(K[tok_offset + j]);
            s_q[j] = q_val > 0.0f ? (q_val + 1.0f) : expf(q_val);
            const float k_elu = k_val > 0.0f ? (k_val + 1.0f) : expf(k_val);
            s_k[j] = k_elu * inv_sqrt_d;
            v_j = to_float(V[tok_offset + j]);
        }
        __syncthreads();

        // 1. Recurrent update S_col[i] = d_val * S_col[i] + s_k[i] * v_j
        // and compute numerator: num = sum_i s_q[i] * S_col[i]
        float num = 0.0f;
        #pragma unroll
        for (int i = 0; i < D && i < MAX_D; ++i) {
            S_col[i] = d_val * S_col[i] + s_k[i] * v_j;
            num += s_q[i] * S_col[i];
        }

        // 2. Thread 0 updates Z and calculates inverse denominator
        if (j == 0) {
            float den = 0.0f;
            #pragma unroll
            for (int i = 0; i < D && i < MAX_D; ++i) {
                s_z[i] = d_val * s_z[i] + s_k[i];
                den += s_q[i] * s_z[i];
            }
            if (den < 1e-5f) den = 1e-5f;
            s_inv_den = 1.0f / den;
        }
        __syncthreads();

        if (j < D) {
            out[tok_offset + j] = from_float<scalar_t>(num * s_inv_den);
        }
        __syncthreads();
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor fms_scan_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& decay
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D = q.size(3);

    auto out = torch::empty_like(q);
    const float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(D));

    auto decay_1d = decay.reshape({-1}).contiguous().to(torch::kFloat32);

    const dim3 grid(1, B * H);
    const int threads = (D <= 64) ? 64 : 128;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "fms_scan_fwd", ([&] {
            if (D <= 32) {
                neuralops::cuda::fms_scan_fwd_kernel<scalar_t, 32><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    decay_1d.data_ptr<float>(),
                    B, H, L, D, inv_sqrt_d
                );
            } else if (D <= 64) {
                neuralops::cuda::fms_scan_fwd_kernel<scalar_t, 64><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    decay_1d.data_ptr<float>(),
                    B, H, L, D, inv_sqrt_d
                );
            } else {
                neuralops::cuda::fms_scan_fwd_kernel<scalar_t, 128><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    decay_1d.data_ptr<float>(),
                    B, H, L, D, inv_sqrt_d
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
