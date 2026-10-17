#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Entropy Row Reduction Kernel:
// Form: p_i = exp(scores_i - max(scores)) / \sum exp(scores_j - max(scores))
//       H = -\sum p_i \log(p_i)
//
// Hardware-level optimization:
// 1. Thread Mapping: 1 block per row, cooperative warp-level / block-level reductions.
// 2. Warp Shuffles: Utilizes block_reduce_max and block_reduce_sum without serial atomic collisions.
// 3. Online Numerical Stability: Subtracts row max before computing exponentials.
// 4. Output: Single scalar entropy written per row in a single global transaction.
template <typename scalar_t>
__global__ void __launch_bounds__(256) entropy_row_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ scores,
    const int rows,
    const int L
) {
    const int r = blockIdx.x;
    if (r >= rows) return;

    const int tid = threadIdx.x;
    const int row_offset = r * L;

    // Pass 1: Find Row Maximum
    float local_max = -1e30f;
    for (int i = tid; i < L; i += blockDim.x) {
        const float v = to_float(scores[row_offset + i]);
        local_max = fmaxf(local_max, v);
    }
    const float m = block_reduce_max(local_max);

    // Pass 2: Compute Softmax Denominator
    float local_sum = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        local_sum += expf(to_float(scores[row_offset + i]) - m);
    }
    const float total_sum = block_reduce_sum(local_sum);
    const float inv_denom = 1.0f / fmaxf(total_sum, 1e-12f);

    // Pass 3: Accumulate Entropy -p * log(p)
    float local_ent = 0.0f;
    for (int i = tid; i < L; i += blockDim.x) {
        const float p = expf(to_float(scores[row_offset + i]) - m) * inv_denom;
        if (p > 1e-12f) {
            local_ent = fmaf(-p, logf(p), local_ent);
        }
    }
    const float total_ent = block_reduce_sum(local_ent);

    if (tid == 0) {
        out[r] = from_float<scalar_t>(total_ent);
    }
}

}} // namespace neuralops::cuda

torch::Tensor entropy_row_reduction_fwd(const torch::Tensor& scores) {
    CHECK_INPUT(scores);

    const int L = scores.size(-1);
    const int rows = scores.numel() / L;
    auto out = torch::empty({rows}, scores.options());
    if (rows == 0 || L == 0) return out;

    int threads = std::min(256, ((L + 31) / 32) * 32);
    if (threads < 32) threads = 32;

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto sc_c = scores.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        scores.scalar_type(), "entropy_row_reduction_fwd", ([&] {
            neuralops::cuda::entropy_row_kernel<scalar_t><<<rows, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                sc_c.data_ptr<scalar_t>(),
                rows,
                L
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
