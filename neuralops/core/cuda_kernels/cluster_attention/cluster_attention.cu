#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Cluster Attention Forward Kernel:
// Form: S_{i, c} = \frac{1}{\sqrt{D}} \sum_{d=0}^{D-1} Q_{i, d} \cdot \text{Centroid}_{c, d}
//       w_{i, c} = \text{Softmax}_c(S_{i, c})
//       out_{i, d} = \sum_{c=0}^{K-1} w_{i, c} \cdot V_{c, d}
//
// Hardware-level optimization:
// 1. True Vector Dot-Product: Evaluates inner products over full head dimension D using online softmax.
// 2. Online Softmax Tracking: Tracks running maximum m_i and normalizer l_i across cluster centroids K.
// 3. Grid/Block mapping: dim3 grid(ceil(N / 32), B * H) with 32 threads/block.
// 4. Strict FP32 intermediate accumulation to prevent numerical overflow and underflow.
template <typename scalar_t>
__global__ void __launch_bounds__(128) cluster_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ centroids,
    const scalar_t* __restrict__ v,
    const int* __restrict__ cluster_indices,
    const int B,
    const int H,
    const int N,
    const int D,
    const int K
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int q_offset = bh * (N * D) + i * D;
    const int c_base_offset = bh * (K * D);
    const float scale = rsqrtf(static_cast<float>(D));

    // Pass 1: Online Softmax over K centroids
    float m_prev = -1e30f;
    float d_prev = 0.0f;

    for (int c = 0; c < K; ++c) {
        float dot = 0.0f;
        const int c_offset = c_base_offset + c * D;
        for (int d = 0; d < D; ++d) {
            dot += to_float(q[q_offset + d]) * to_float(centroids[c_offset + d]);
        }
        dot *= scale;

        const float m_curr = fmaxf(m_prev, dot);
        const float exp_curr = expf(dot - m_curr);
        d_prev = d_prev * expf(m_prev - m_curr) + exp_curr;
        m_prev = m_curr;
    }

    const float inv_d = (d_prev > 1e-6f) ? (1.0f / d_prev) : 0.0f;

    // Pass 2: Weighted combination of value vectors V
    for (int d = 0; d < D; ++d) {
        float val_acc = 0.0f;
        for (int c = 0; c < K; ++c) {
            float dot = 0.0f;
            const int c_offset = c_base_offset + c * D;
            for (int dd = 0; dd < D; ++dd) {
                dot += to_float(q[q_offset + dd]) * to_float(centroids[c_offset + dd]);
            }
            dot *= scale;
            const float weight = expf(dot - m_prev) * inv_d;
            val_acc += weight * to_float(v[c_base_offset + c * D + d]);
        }
        out[q_offset + d] = from_float<scalar_t>(val_acc);
    }
}

}} // namespace neuralops::cuda

torch::Tensor cluster_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& cluster_centroids,
    const torch::Tensor& v,
    const torch::Tensor& cluster_indices
) {
    CHECK_INPUT(q);
    CHECK_INPUT(cluster_centroids);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, D]");
    TORCH_CHECK(cluster_centroids.dim() == 4, "cluster_centroids must be 4D [B, H, K, D]");
    TORCH_CHECK(v.dim() == 4, "v must be 4D [B, H, K, D]");

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    const int K = cluster_centroids.size(2);

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || N == 0 || D == 0 || K == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (N + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto c_c = cluster_centroids.contiguous();
    auto v_c = v.contiguous();
    auto idx_c = cluster_indices.defined() ? cluster_indices.contiguous() : torch::empty(0);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "cluster_attn_fwd", ([&] {
            neuralops::cuda::cluster_attn_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                c_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                idx_c.defined() ? idx_c.data_ptr<int>() : nullptr,
                B, H, N, D, K
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
