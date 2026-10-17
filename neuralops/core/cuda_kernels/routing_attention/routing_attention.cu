#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Routing Attention Forward Kernel:
// Form: Attends strictly to tokens routed by positive routing weights route_weights[j] > 0.
//
// Hardware-level optimization:
// 1. True Vector Dot-Products across D using Online Softmax.
// 2. Fused routing weight modulation: w_{i, j} = exp(Q_i K_j^T / sqrt(D)) * route_weights_j.
// 3. Strict FP32 intermediate accumulation to avoid precision overflow.
template <typename scalar_t>
__global__ void __launch_bounds__(128) routing_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ route_weights,
    const int B,
    const int H,
    const int N,
    const int D,
    const int top_k
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int base_offset = bh * (N * D);
    const int qi_offset = base_offset + i * D;
    const float scale = rsqrtf(static_cast<float>(D));

    // Pass 1: Online Softmax across routed candidates
    float m_prev = -1e30f;
    float d_prev = 0.0f;

    for (int j = 0; j < N; ++j) {
        const float r_ij = to_float(route_weights[bh * N + j]);
        if (r_ij <= 0.0f) continue;

        const int kj_offset = base_offset + j * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += to_float(q[qi_offset + d]) * to_float(k[kj_offset + d]);
        }
        dot *= scale;

        const float m_curr = fmaxf(m_prev, dot);
        d_prev = d_prev * expf(m_prev - m_curr) + expf(dot - m_curr) * r_ij;
        m_prev = m_curr;
    }

    const float inv_d = (d_prev > 1e-6f) ? (1.0f / d_prev) : 0.0f;

    // Pass 2: Accumulate weighted values
    for (int d = 0; d < D; ++d) {
        float val_acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            const float r_ij = to_float(route_weights[bh * N + j]);
            if (r_ij <= 0.0f) continue;

            const int kj_offset = base_offset + j * D;
            float dot = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                dot += to_float(q[qi_offset + dd]) * to_float(k[kj_offset + dd]);
            }
            dot *= scale;

            const float weight = expf(dot - m_prev) * r_ij * inv_d;
            val_acc += weight * to_float(v[kj_offset + d]);
        }
        out[qi_offset + d] = from_float<scalar_t>(val_acc);
    }
}

}} // namespace neuralops::cuda

torch::Tensor routing_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& route_weights,
    int top_k
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(route_weights);

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, D]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape mismatch");

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (N + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto r_c = route_weights.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "routing_attn_fwd", ([&] {
            neuralops::cuda::routing_attn_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                r_c.data_ptr<scalar_t>(),
                B, H, N, D, top_k
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
