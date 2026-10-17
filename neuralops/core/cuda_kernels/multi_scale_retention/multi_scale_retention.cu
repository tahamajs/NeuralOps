#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Multi-Scale Retention (RetNet) Forward Scan: S_t = gamma_h * S_{t-1} + k_t * v_t^T, y_t = q_t * S_t
template <typename scalar_t, int MAX_D>
__launch_bounds__(32)
__global__ void multi_scale_retention_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const float* __restrict__ gammas,
    const int B,
    const int H,
    const int L,
    const int D
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int h = bh % H;
    const float gamma = gammas[h];

    float S[MAX_D][MAX_D];
    #pragma unroll 4
    for (int i = 0; i < MAX_D; ++i) {
        #pragma unroll 4
        for (int j = 0; j < MAX_D; ++j) {
            S[i][j] = 0.0f;
        }
    }

    const int head_offset = bh * (L * D);

    for (int t = 0; t < L; ++t) {
        const int tok_offset = head_offset + t * D;

        // 1. S_t = gamma * S_{t-1} + k_t * v_t^T
        #pragma unroll 4
        for (int i = 0; i < D && i < MAX_D; ++i) {
            const float k_val = to_float(K[tok_offset + i]);
            #pragma unroll 4
            for (int j = 0; j < D && j < MAX_D; ++j) {
                S[i][j] = gamma * S[i][j] + k_val * to_float(V[tok_offset + j]);
            }
        }

        // 2. y_t = q_t * S_t
        #pragma unroll 4
        for (int j = 0; j < D && j < MAX_D; ++j) {
            float y_acc = 0.0f;
            #pragma unroll 4
            for (int i = 0; i < D && i < MAX_D; ++i) {
                y_acc += to_float(Q[tok_offset + i]) * S[i][j];
            }
            out[tok_offset + j] = from_float<scalar_t>(y_acc);
        }
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor multi_scale_retention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gammas
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D = q.size(3);

    auto out = torch::empty_like(q);

    const dim3 grid(1, B * H);
    const int threads = 32;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto g_c = gammas.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "multi_scale_retention_fwd", ([&] {
            if (D <= 16) {
                neuralops::cuda::multi_scale_retention_fwd_kernel<scalar_t, 16><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    g_c.data_ptr<float>(),
                    B, H, L, D
                );
            } else if (D <= 32) {
                neuralops::cuda::multi_scale_retention_fwd_kernel<scalar_t, 32><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    g_c.data_ptr<float>(),
                    B, H, L, D
                );
            } else if (D <= 64) {
                neuralops::cuda::multi_scale_retention_fwd_kernel<scalar_t, 64><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    g_c.data_ptr<float>(),
                    B, H, L, D
                );
            } else {
                neuralops::cuda::multi_scale_retention_fwd_kernel<scalar_t, 128><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    g_c.data_ptr<float>(),
                    B, H, L, D
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
