#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused Factorised Matrix State (FMS) Recurrent Scan Kernel
template <typename scalar_t, int MAX_D>
__launch_bounds__(32)
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
    const int bh = blockIdx.x;
    if (bh >= B * H) return;

    const int h = bh % H;
    const float d_val = decay[h];

    // State S: [MAX_D, MAX_D], State Z: [MAX_D]
    float S[MAX_D][MAX_D];
    float Z[MAX_D];

    #pragma unroll
    for (int i = 0; i < MAX_D; ++i) {
        Z[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < MAX_D; ++j) {
            S[i][j] = 0.0f;
        }
    }

    const int head_offset = bh * (L * D);

    for (int t = 0; t < L; ++t) {
        const int tok_offset = head_offset + t * D;

        // 1. Feature map ELU + 1
        float q_f[MAX_D];
        float k_f[MAX_D];
        float v_f[MAX_D];

        for (int d = 0; d < D && d < MAX_D; ++d) {
            const float q_val = to_float(Q[tok_offset + d]);
            const float k_val = to_float(K[tok_offset + d]);
            q_f[d] = q_val > 0.0f ? (q_val + 1.0f) : expf(q_val);
            const float k_elu = k_val > 0.0f ? (k_val + 1.0f) : expf(k_val);
            k_f[d] = k_elu * inv_sqrt_d;
            v_f[d] = to_float(V[tok_offset + d]);
        }

        // 2. Recurrent update: S_t = decay * S_{t-1} + k_t @ v_t
        for (int i = 0; i < D && i < MAX_D; ++i) {
            Z[i] = d_val * Z[i] + k_f[i];
            for (int j = 0; j < D && j < MAX_D; ++j) {
                S[i][j] = d_val * S[i][j] + k_f[i] * v_f[j];
            }
        }

        // 3. Compute denominator: sum_i q_f[i] * Z[i]
        float den = 0.0f;
        for (int i = 0; i < D && i < MAX_D; ++i) {
            den += q_f[i] * Z[i];
        }
        if (den < 1e-5f) den = 1e-5f;
        const float inv_den = 1.0f / den;

        // 4. Compute numerator and output: num_j = sum_i q_f[i] * S[i, j]
        for (int j = 0; j < D && j < MAX_D; ++j) {
            float num = 0.0f;
            for (int i = 0; i < D && i < MAX_D; ++i) {
                num += q_f[i] * S[i][j];
            }
            out[tok_offset + j] = from_float<scalar_t>(num * inv_den);
        }
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

    const int blocks = B * H;
    const int threads = 1;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "fms_scan_fwd", ([&] {
            if (D <= 16) {
                neuralops::cuda::fms_scan_fwd_kernel<scalar_t, 16><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    decay_1d.data_ptr<float>(),
                    B, H, L, D, inv_sqrt_d
                );
            } else if (D <= 32) {
                neuralops::cuda::fms_scan_fwd_kernel<scalar_t, 32><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    decay_1d.data_ptr<float>(),
                    B, H, L, D, inv_sqrt_d
                );
            } else if (D <= 64) {
                neuralops::cuda::fms_scan_fwd_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    decay_1d.data_ptr<float>(),
                    B, H, L, D, inv_sqrt_d
                );
            } else {
                neuralops::cuda::fms_scan_fwd_kernel<scalar_t, 128><<<blocks, threads, 0, stream>>>(
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
