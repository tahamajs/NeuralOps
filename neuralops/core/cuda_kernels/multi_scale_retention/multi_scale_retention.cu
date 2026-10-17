#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Multi-Scale Retention (RetNet) Forward Scan: S_t = gamma_h * S_{t-1} + k_t * v_t^T, y_t = q_t * S_t
template <typename scalar_t, int MAX_D>
__launch_bounds__(128)
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
    const int j = threadIdx.x;

    float S_col[MAX_D];
    #pragma unroll
    for (int i = 0; i < MAX_D; ++i) {
        S_col[i] = 0.0f;
    }

    __shared__ float s_k[MAX_D];
    __shared__ float s_q[MAX_D];

    const int head_offset = bh * (L * D);

    for (int t = 0; t < L; ++t) {
        const int tok_offset = head_offset + t * D;

        if (j < D && j < MAX_D) {
            s_k[j] = to_float(K[tok_offset + j]);
            s_q[j] = to_float(Q[tok_offset + j]);
        }
        __syncthreads();

        const float v_j = (j < D) ? to_float(V[tok_offset + j]) : 0.0f;

        float y_acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < D && i < MAX_D; ++i) {
            S_col[i] = gamma * S_col[i] + s_k[i] * v_j;
            y_acc += s_q[i] * S_col[i];
        }

        if (j < D) {
            out[tok_offset + j] = from_float<scalar_t>(y_acc);
        }
        __syncthreads();
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
    const int threads = (D <= 64) ? 64 : 128;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto g_c = gammas.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "multi_scale_retention_fwd", ([&] {
            if (D <= 32) {
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
