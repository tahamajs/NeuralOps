#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Speculative draft verify kernel
// draft_tokens: [B, K], target_tokens: [B, K]
// accept_mask: [B, K] (bool/uint8), accept_count: [B] (int32)
__global__ void __launch_bounds__(32) speculative_verify_kernel(
    uint8_t* __restrict__ accept_mask,
    int32_t* __restrict__ accept_count,
    const int64_t* __restrict__ draft_tokens,
    const int64_t* __restrict__ target_tokens,
    const int B,
    const int K
) {
    const int b = blockIdx.x;
    if (b >= B) return;

    if (threadIdx.x == 0) {
        bool still_accepted = true;
        int count = 0;

        for (int k = 0; k < K; ++k) {
            const int idx = b * K + k;
            if (still_accepted && draft_tokens[idx] == target_tokens[idx]) {
                accept_mask[idx] = 1;
                count++;
            } else {
                still_accepted = false;
                accept_mask[idx] = 0;
            }
        }

        accept_count[b] = count;
    }
}

} // namespace cuda
} // namespace neuralops

std::vector<torch::Tensor> speculative_verify_fwd(
    const torch::Tensor& draft_tokens,
    const torch::Tensor& target_tokens
) {
    CHECK_INPUT(draft_tokens);
    CHECK_INPUT(target_tokens);

    const int B = draft_tokens.size(0);
    const int K = draft_tokens.size(1);

    auto accept_mask = torch::empty({B, K}, draft_tokens.options().dtype(torch::kUInt8));
    auto accept_count = torch::empty({B}, draft_tokens.options().dtype(torch::kInt32));
    if (B == 0 || K == 0) {
        return {accept_mask.to(torch::kBool), accept_count};
    }

    const int blocks = B;
    const int threads = 32;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto d_c = draft_tokens.contiguous().to(torch::kInt64);
    auto t_c = target_tokens.contiguous().to(torch::kInt64);

    neuralops::cuda::speculative_verify_kernel<<<blocks, threads, 0, stream>>>(
        accept_mask.data_ptr<uint8_t>(),
        accept_count.data_ptr<int32_t>(),
        d_c.data_ptr<int64_t>(),
        t_c.data_ptr<int64_t>(),
        B, K
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {accept_mask.to(torch::kBool), accept_count};
}
