#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Entmax15 (alpha=1.5) Bisection Root Finding Kernel:
// Form: p_i = [ (x_i - \tau) / 2 ]_+^2
// Root condition: \sum_i p_i = 1.0
//
// Hardware-level optimization:
// 1. Shared-memory row cache: Eliminates redundant global memory loads across all bisection iterations.
// 2. 128-bit float4 vectorized score loading into shared memory.
// 3. Fast block-level reductions (max and sum) with zero inter-block synchronization overhead.
// 4. Strict FP32 intermediate precision and branchless fmaxf.
template <typename scalar_t>
__global__ void __launch_bounds__(256) entmax15_shared_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ scores,
    const int rows,
    const int N,
    const int n_iter
) {
    const int r = blockIdx.x;
    if (r >= rows) return;

    const int tid = threadIdx.x;
    const int row_offset = r * N;

    extern __shared__ float s_scores[];

    const int num_vec = N / 4;
    float local_max = -1e30f;

    // Vectorized load into shared memory and local max computation
    for (int i = tid; i < num_vec; i += blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 in4 = *reinterpret_cast<const float4*>(scores + row_offset + i * 4);
            s_scores[i * 4 + 0] = in4.x;
            s_scores[i * 4 + 1] = in4.y;
            s_scores[i * 4 + 2] = in4.z;
            s_scores[i * 4 + 3] = in4.w;

            local_max = fmaxf(local_max, fmaxf(fmaxf(in4.x, in4.y), fmaxf(in4.z, in4.w)));
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float val = to_float(scores[row_offset + i * 4 + k]);
                s_scores[i * 4 + k] = val;
                local_max = fmaxf(local_max, val);
            }
        }
    }

    const int rem_start = num_vec * 4;
    for (int i = rem_start + tid; i < N; i += blockDim.x) {
        const float val = to_float(scores[row_offset + i]);
        s_scores[i] = val;
        local_max = fmaxf(local_max, val);
    }
    __syncthreads();

    const float m = block_reduce_max(local_max);

    // Initial bounds for bisection
    float low = m - 2.0f;
    float high = m;

    // Warp/Block Coordinated Bisection using shared memory
    for (int it = 0; it < n_iter; ++it) {
        const float mid = 0.5f * (low + high);
        float local_sum = 0.0f;

        for (int i = tid; i < N; i += blockDim.x) {
            const float d = (s_scores[i] - mid) * 0.5f;
            if (d > 0.0f) {
                local_sum = fmaf(d, d, local_sum);
            }
        }

        const float total_sum = block_reduce_sum(local_sum);
        if (total_sum > 1.0f) {
            low = mid;
        } else {
            high = mid;
        }
        __syncthreads();
    }

    // Final normalization
    const float final_tau = 0.5f * (low + high);
    float final_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        const float d = (s_scores[i] - final_tau) * 0.5f;
        const float p = (d > 0.0f) ? (d * d) : 0.0f;
        final_sum += p;
    }
    const float total_final = block_reduce_sum(final_sum);
    const float inv_denom = 1.0f / fmaxf(total_final, 1e-9f);

    for (int i = tid; i < num_vec; i += blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            float4 out4;
            #define EVAL_P(idx_k, out_elem) { \
                const float d = (s_scores[idx_k] - final_tau) * 0.5f; \
                out_elem = (d > 0.0f) ? ((d * d) * inv_denom) : 0.0f; \
            }
            EVAL_P(i * 4 + 0, out4.x);
            EVAL_P(i * 4 + 1, out4.y);
            EVAL_P(i * 4 + 2, out4.z);
            EVAL_P(i * 4 + 3, out4.w);
            #undef EVAL_P

            *reinterpret_cast<float4*>(out + row_offset + i * 4) = out4;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float d = (s_scores[i * 4 + k] - final_tau) * 0.5f;
                const float p = (d > 0.0f) ? ((d * d) * inv_denom) : 0.0f;
                out[row_offset + i * 4 + k] = from_float<scalar_t>(p);
            }
        }
    }

    for (int i = rem_start + tid; i < N; i += blockDim.x) {
        const float d = (s_scores[i] - final_tau) * 0.5f;
        const float p = (d > 0.0f) ? ((d * d) * inv_denom) : 0.0f;
        out[row_offset + i] = from_float<scalar_t>(p);
    }
}

}} // namespace neuralops::cuda

torch::Tensor entmax15_fwd(const torch::Tensor& scores, int n_iter) {
    CHECK_INPUT(scores);

    auto orig_shape = scores.sizes();
    const int N = orig_shape.back();
    const int rows = scores.numel() / N;
    auto out = torch::empty_like(scores);
    if (rows == 0 || N == 0) return out;

    int threads = std::min(256, ((N + 31) / 32) * 32);
    if (threads < 32) threads = 32;

    const size_t shared_mem = N * sizeof(float);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto sc_c = scores.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        scores.scalar_type(), "entmax15_fwd", ([&] {
            neuralops::cuda::entmax15_shared_kernel<scalar_t><<<rows, threads, shared_mem, stream>>>(
                out.data_ptr<scalar_t>(),
                sc_c.data_ptr<scalar_t>(),
                rows,
                N,
                n_iter
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
