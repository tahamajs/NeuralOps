#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Linformer Low-Rank Attention Kernel:
// Projects K and V across length N to K_proj, V_proj of length K_proj_len.
// Calculates true Q @ K_proj^T attention scores along D, normalizes via Softmax,
// and computes the weighted combination with V_proj.
// Thread mapping: 2D Grid over (ceil(N / 32), B * H), with Block Dim (32, 1).
template <typename scalar_t>
__global__ void __launch_bounds__(128) linformer_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_proj,
    const scalar_t* __restrict__ v_proj,
    const int B,
    const int H,
    const int N,
    const int K_proj_len,
    const int D
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int q_offset = bh * (N * D) + i * D;
    const int kv_head_offset = bh * (K_proj_len * D);
    const float scale = rsqrtf(static_cast<float>(D));

    // Dynamic / register local buffer for projected key matching
    // For production resilience, we stream over K_proj_len to find max, compute softmax weights,
    // and accumulate into out along D.
    float m_prev = -1e30f;
    float d_prev = 0.0f;
    float scores[128];

    for (int k = 0; k < K_proj_len && k < 128; ++k) {
        float dot = 0.0f;
        const int k_offset = kv_head_offset + k * D;
        for (int d = 0; d < D; ++d) {
            dot += to_float(q[q_offset + d]) * to_float(k_proj[k_offset + d]);
        }
        dot *= scale;
        scores[k] = dot;
        m_prev = fmaxf(m_prev, dot);
    }

    for (int k = 0; k < K_proj_len && k < 128; ++k) {
        float exp_val = expf(scores[k] - m_prev);
        scores[k] = exp_val;
        d_prev += exp_val;
    }

    const float inv_d = (d_prev > 1e-6f) ? (1.0f / d_prev) : 0.0f;
    for (int k = 0; k < K_proj_len && k < 128; ++k) {
        scores[k] *= inv_d;
    }

    // Pass 2: accumulate weighted V_proj
    for (int d = 0; d < D; ++d) {
        float val_acc = 0.0f;
        for (int k = 0; k < K_proj_len && k < 128; ++k) {
            val_acc += scores[k] * to_float(v_proj[kv_head_offset + k * D + d]);
        }
        out[q_offset + d] = from_float<scalar_t>(val_acc);
    }
}

}} // namespace neuralops::cuda

torch::Tensor linformer_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k_proj,
    const torch::Tensor& v_proj
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k_proj);
    CHECK_INPUT(v_proj);

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    const int K_proj_len = k_proj.size(2);
    TORCH_CHECK(K_proj_len <= 128, "linformer_attn_fwd: K_proj_len=", K_proj_len, " > 128 not supported");

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (N + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k_proj.contiguous();
    auto v_c = v_proj.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "linformer_attn_fwd", ([&] {
            neuralops::cuda::linformer_attn_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, N, K_proj_len, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
