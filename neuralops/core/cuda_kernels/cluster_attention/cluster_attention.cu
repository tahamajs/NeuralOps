#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Single-Pass Warp-Per-Query Flash Cluster Attention Forward Kernel:
// Form: S_{i, c} = \frac{1}{\sqrt{D}} \sum_{d=0}^{D-1} Q_{i, d} \cdot \text{Centroid}_{c, d}
//       w_{i, c} = \text{Softmax}_c(S_{i, c})
//       out_{i, d} = \sum_{c=0}^{K-1} w_{i, c} \cdot V_{c, d}
//
// Hardware-level optimization:
// 1. Warp-Per-Query (32 threads/query token): 100% coalesced vector loads across lane and zero register spilling.
// 2. Single-pass fused online softmax rescaling and value accumulation in registers.
// 3. Strict FP32 intermediate accumulation to prevent numerical overflow and underflow.
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

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int i = blockIdx.x * (blockDim.x / 32) + warp_id;
    if (i >= N) return;

    const int q_offset = bh * (N * D) + i * D;
    const int c_base_offset = bh * (K * D);
    const float scale = rsqrtf(static_cast<float>(D));

    // Register cache for query vectors (up to D=128, e.g. 2 elements per lane for D=64, 4 for D=128)
    constexpr int MAX_E = 4;
    float q_reg[MAX_E];

    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        if (d < D) {
            q_reg[e] = to_float(q[q_offset + d]) * scale;
        } else {
            q_reg[e] = 0.0f;
        }
    }

    // Single-pass FlashAttention across K centroids
    float m_prev = -1e30f;
    float d_prev = 0.0f;
    float acc[MAX_E] = {0.0f};

    for (int c = 0; c < K; ++c) {
        float dot_local = 0.0f;
        const int c_offset = c_base_offset + c * D;

        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                dot_local = fmaf(q_reg[e], to_float(centroids[c_offset + d]), dot_local);
            }
        }

        // Warp reduction across 32 lanes
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot_local += __shfl_xor_sync(0xffffffff, dot_local, offset);
        }

        // Online softmax update
        const float m_curr = fmaxf(m_prev, dot_local);
        const float alpha = expf(m_prev - m_curr);
        const float p = expf(dot_local - m_curr);
        d_prev = d_prev * alpha + p;
        m_prev = m_curr;

        // Rescale prior accumulation and add current p * v
        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                const float v_val = to_float(v[c_offset + d]);
                acc[e] = acc[e] * alpha + p * v_val;
            }
        }
    }

    const float inv_d = (d_prev > 1e-6f) ? (1.0f / d_prev) : 0.0f;

    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        if (d < D) {
            out[q_offset + d] = from_float<scalar_t>(acc[e] * inv_d);
        }
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
    TORCH_CHECK(D <= 128, "cluster_attn_fwd: D=", D, " exceeds max 128");

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || N == 0 || D == 0 || K == 0) return out;

    constexpr int THREADS = 128; // 4 warps per block -> 4 queries per block
    constexpr int WARPS_PER_BLOCK = THREADS / 32;
    const int blocks_x = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
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
