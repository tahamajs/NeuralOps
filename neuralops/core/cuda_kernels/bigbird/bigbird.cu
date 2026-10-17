#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// BigBird Sparse Attention Forward Kernel:
// Form: Attends to global tokens, sliding local window tokens, and pseudo-random tokens.
//
// Hardware-level optimization:
// 1. True Vector Dot-Products across D using Online Softmax.
// 2. Exact sparse mask condition: is_global || is_local || is_random.
// 3. Strict FP32 intermediate accumulation to avoid precision overflow.
template <typename scalar_t>
__global__ void __launch_bounds__(128) bigbird_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int N,
    const int D,
    const int window_size,
    const int num_global_tokens,
    const int num_random_blocks
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int base_offset = bh * (N * D);
    const int qi_offset = base_offset + i * D;
    const float scale = rsqrtf(static_cast<float>(D));
    const int rand_mod = max(1, N / max(1, num_random_blocks));

    // Pass 1: Online Softmax across sparse pattern
    float m_prev = -1e30f;
    float d_prev = 0.0f;

    for (int j = 0; j < N; ++j) {
        const bool is_global = (j < num_global_tokens) || (i < num_global_tokens);
        const bool is_local = abs(i - j) <= window_size;
        const bool is_random = ((i * 37 + j * 17) % rand_mod) == 0;
        if (!is_global && !is_local && !is_random) continue;

        const int kj_offset = base_offset + j * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += to_float(q[qi_offset + d]) * to_float(k[kj_offset + d]);
        }
        dot *= scale;

        const float m_curr = fmaxf(m_prev, dot);
        d_prev = d_prev * expf(m_prev - m_curr) + expf(dot - m_curr);
        m_prev = m_curr;
    }

    const float inv_d = (d_prev > 1e-6f) ? (1.0f / d_prev) : 0.0f;

    // Pass 2: Accumulate weighted values
    for (int d = 0; d < D; ++d) {
        float val_acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            const bool is_global = (j < num_global_tokens) || (i < num_global_tokens);
            const bool is_local = abs(i - j) <= window_size;
            const bool is_random = ((i * 37 + j * 17) % rand_mod) == 0;
            if (!is_global && !is_local && !is_random) continue;

            const int kj_offset = base_offset + j * D;
            float dot = 0.0f;
            for (int dd = 0; dd < D; ++dd) {
                dot += to_float(q[qi_offset + dd]) * to_float(k[kj_offset + dd]);
            }
            dot *= scale;

            const float weight = expf(dot - m_prev) * inv_d;
            val_acc += weight * to_float(v[kj_offset + d]);
        }
        out[qi_offset + d] = from_float<scalar_t>(val_acc);
    }
}

}} // namespace neuralops::cuda

torch::Tensor bigbird_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int window_size,
    int num_global_tokens,
    int num_random_blocks
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

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

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "bigbird_fwd", ([&] {
            neuralops::cuda::bigbird_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, N, D, window_size, num_global_tokens, num_random_blocks
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
