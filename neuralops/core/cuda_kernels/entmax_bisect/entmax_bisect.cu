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
// 1. Warp/Block Coordinated Bisection: Threads evaluate partial sums cooperatively.
// 2. Uniform Tau Broadcast: Tau is broadcast to all threads via warp primitives; no thread divergence.
// 3. Fast Block Reduction: Uses block_reduce_sum without serial global atomic collisions.
// 4. Strict FP32 intermediate precision.
template <typename scalar_t>
__global__ void __launch_bounds__(256) entmax15_kernel(
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

    // Step 1: Find Row Max
    float local_max = -1e30f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, to_float(scores[row_offset + i]));
    }
    const float m = block_reduce_max(local_max);

    // Initial bounds for bisection
    float low = m - 2.0f;
    float high = m;

    // Step 2: Warp/Block Coordinated Bisection
    for (int it = 0; it < n_iter; ++it) {
        const float mid = 0.5f * (low + high);
        float local_sum = 0.0f;

        for (int i = tid; i < N; i += blockDim.x) {
            const float d = (to_float(scores[row_offset + i]) - mid) * 0.5f;
            if (d > 0.0f) {
                local_sum += d * d;
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

    // Step 3: Compute final probabilities
    const float final_tau = 0.5f * (low + high);
    float final_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        const float d = (to_float(scores[row_offset + i]) - final_tau) * 0.5f;
        const float p = (d > 0.0f) ? (d * d) : 0.0f;
        final_sum += p;
    }
    const float total_final = block_reduce_sum(final_sum);
    const float inv_denom = 1.0f / fmaxf(total_final, 1e-9f);

    for (int i = tid; i < N; i += blockDim.x) {
        const float d = (to_float(scores[row_offset + i]) - final_tau) * 0.5f;
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

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto sc_c = scores.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        scores.scalar_type(), "entmax15_fwd", ([&] {
            neuralops::cuda::entmax15_kernel<scalar_t><<<rows, threads, 0, stream>>>(
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
