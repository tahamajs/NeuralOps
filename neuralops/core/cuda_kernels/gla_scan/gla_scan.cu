#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// GLA Scan: S_t = alpha_t * S_{t-1} + k_t * v_t^T, y_t = q_t * S_t
template <typename scalar_t, int MAX_D>
__launch_bounds__(32)
__global__ void gla_scan_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const scalar_t* __restrict__ alpha,
    const int B,
    const int H,
    const int L,
    const int D,
    const float scale,
    const bool has_alpha
) {
    const int bh = blockIdx.x;
    if (bh >= B * H) return;

    float S[MAX_D][MAX_D];
    #pragma unroll
    for (int i = 0; i < MAX_D; ++i) {
        #pragma unroll
        for (int j = 0; j < MAX_D; ++j) {
            S[i][j] = 0.0f;
        }
    }

    const int head_offset = bh * (L * D);

    for (int t = 0; t < L; ++t) {
        const int tok_offset = head_offset + t * D;

        // Optional decay factor
        float a_val = 1.0f;
        if (has_alpha && alpha != nullptr) {
            a_val = to_float(alpha[bh * L + t]);
        }

        // 1. Decay state and accumulate S_t = a_val * S_{t-1} + k_t * v_t^T
        for (int i = 0; i < D && i < MAX_D; ++i) {
            const float k_val = to_float(K[tok_offset + i]);
            for (int j = 0; j < D && j < MAX_D; ++j) {
                S[i][j] = a_val * S[i][j] + k_val * to_float(V[tok_offset + j]);
            }
        }

        // 2. Compute y_t = (q_t * S_t) * scale
        for (int j = 0; j < D && j < MAX_D; ++j) {
            float y_acc = 0.0f;
            for (int i = 0; i < D && i < MAX_D; ++i) {
                y_acc += to_float(Q[tok_offset + i]) * S[i][j];
            }
            out[tok_offset + j] = from_float<scalar_t>(y_acc * scale);
        }
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor gla_scan_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& alpha
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D = q.size(3);

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf(static_cast<float>(D));
    const bool has_alpha = alpha.has_value() && alpha.value().defined();

    const int blocks = B * H;
    const int threads = 1;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "gla_scan_fwd", ([&] {
            if (D <= 16) {
                neuralops::cuda::gla_scan_fwd_kernel<scalar_t, 16><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    has_alpha ? alpha.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    B, H, L, D, scale, has_alpha
                );
            } else if (D <= 32) {
                neuralops::cuda::gla_scan_fwd_kernel<scalar_t, 32><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    has_alpha ? alpha.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    B, H, L, D, scale, has_alpha
                );
            } else if (D <= 64) {
                neuralops::cuda::gla_scan_fwd_kernel<scalar_t, 64><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    has_alpha ? alpha.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    B, H, L, D, scale, has_alpha
                );
            } else {
                neuralops::cuda::gla_scan_fwd_kernel<scalar_t, 128><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    has_alpha ? alpha.value().contiguous().data_ptr<scalar_t>() : nullptr,
                    B, H, L, D, scale, has_alpha
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
