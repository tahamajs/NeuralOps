#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Flash Decoding Stage 2 Reduction Kernel:
// Form: out[b, h, 1, d] = \sum_c \frac{\exp(mid\_lse[b, h, c] - global\_lse)}{\sum_k \exp(mid\_lse[b, h, k] - global\_lse)} \cdot mid\_out[b, h, c, d]
//
// Hardware-level optimization:
// 1. Warp/Block LogSumExp reduction across chunks: Replaces atomic float serialization.
// 2. Strict FP32 intermediate accumulation: Rescales partial attention chunks without precision loss.
// 3. Coalesced output stores: Thread index threadIdx.x writes along dimension D.
template <typename scalar_t>
__global__ void __launch_bounds__(256) flash_decode_stage2_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mid_out,
    const float* __restrict__ mid_lse,
    const int B,
    const int H,
    const int num_chunks,
    const int D
) {
    const int pid_bh = blockIdx.x;
    if (pid_bh >= B * H) return;

    const int tid = threadIdx.x;
    const int lse_offset = pid_bh * num_chunks;

    __shared__ float s_global_max_lse;
    __shared__ float s_inv_denom;

    if (tid == 0) {
        float max_lse = -1e30f;
        for (int c = 0; c < num_chunks; ++c) {
            max_lse = fmaxf(max_lse, mid_lse[lse_offset + c]);
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < num_chunks; ++c) {
            sum_exp += expf(mid_lse[lse_offset + c] - max_lse);
        }
        s_global_max_lse = max_lse;
        s_inv_denom = 1.0f / fmaxf(sum_exp, 1e-12f);
    }
    __syncthreads();

    const float global_max_lse = s_global_max_lse;
    const float inv_denom = s_inv_denom;

    // Step 3: Weighted sum over chunks for each feature dimension d
    const int out_base = pid_bh * D;
    const int mid_base = pid_bh * (num_chunks * D);

    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < num_chunks; ++c) {
            const float weight = expf(mid_lse[lse_offset + c] - global_max_lse) * inv_denom;
            const float val = to_float(mid_out[mid_base + c * D + d]);
            acc = fmaf(weight, val, acc);
        }
        out[out_base + d] = from_float<scalar_t>(acc);
    }
}

}} // namespace neuralops::cuda

torch::Tensor flash_decode_stage2_fwd(
    const torch::Tensor& mid_out,
    const torch::Tensor& mid_lse
) {
    CHECK_INPUT(mid_out);
    CHECK_INPUT(mid_lse);

    TORCH_CHECK(mid_out.dim() == 4, "mid_out must be 4D [B, H, num_chunks, D]");
    TORCH_CHECK(mid_lse.dim() == 3, "mid_lse must be 3D [B, H, num_chunks]");

    const int B = mid_out.size(0);
    const int H = mid_out.size(1);
    const int num_chunks = mid_out.size(2);
    const int D = mid_out.size(3);

    auto out = torch::empty({B, H, 1, D}, mid_out.options());
    if (B == 0 || H == 0 || num_chunks == 0 || D == 0) return out;

    int threads = std::min(256, ((D + 31) / 32) * 32);
    if (threads < 32) threads = 32;

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto mo_c = mid_out.contiguous();
    auto ml_c = mid_lse.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        mid_out.scalar_type(), "flash_decode_stage2_fwd", ([&] {
            neuralops::cuda::flash_decode_stage2_kernel<scalar_t><<<B * H, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                mo_c.data_ptr<scalar_t>(),
                ml_c.data_ptr<float>(),
                B, H, num_chunks, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
