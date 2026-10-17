#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(512) aegis_gate_norm_2b_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ b0,
    const scalar_t* __restrict__ b1,
    const float* __restrict__ weights,
    const int total_tokens,
    const int D,
    const float eps
) {
    const int pid = blockIdx.x;
    if (pid >= total_tokens) return;
    const int tid = threadIdx.x;
    const float raw_w0 = weights[pid * 2 + 0];
    const float raw_w1 = weights[pid * 2 + 1];
    const float max_w = fmaxf(raw_w0, raw_w1);
    const float exp0 = expf(raw_w0 - max_w);
    const float exp1 = expf(raw_w1 - max_w);
    const float inv_sum = 1.0f / (exp0 + exp1 + 1e-8f);
    const float w0 = exp0 * inv_sum;
    const float w1 = exp1 * inv_sum;

    float sq0 = 0.0f, sq1 = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float v0 = to_float(b0[pid * D + i]);
        float v1 = to_float(b1[pid * D + i]);
        sq0 += v0 * v0;
        sq1 += v1 * v1;
    }

    sq0 = block_reduce_sum(sq0);
    sq1 = block_reduce_sum(sq1);

    __shared__ float s_inv0, s_inv1;
    if (tid == 0) {
        s_inv0 = rsqrtf(sq0 / (float)D + eps);
        s_inv1 = rsqrtf(sq1 / (float)D + eps);
    }
    __syncthreads();

    const float inv_rms0 = s_inv0;
    const float inv_rms1 = s_inv1;

    for (int i = tid; i < D; i += blockDim.x) {
        float n0 = to_float(b0[pid * D + i]) * inv_rms0;
        float n1 = to_float(b1[pid * D + i]) * inv_rms1;
        float res = n0 * w0 + n1 * w1;
        out[pid * D + i] = from_float<scalar_t>(res);
    }
}

}} // namespace neuralops::cuda

torch::Tensor aegis_gate_norm_2branch_fwd(
    const torch::Tensor& b0,
    const torch::Tensor& b1,
    const torch::Tensor& weights,
    float eps
) {
    CHECK_INPUT(b0);
    CHECK_INPUT(b1);
    CHECK_INPUT(weights);

    const int D = b0.size(-1);
    const int total_tokens = b0.numel() / D;
    auto out = torch::empty_like(b0);
    if (total_tokens == 0 || D == 0) return out;

    int threads = std::min(512, ((D + 31) / 32) * 32);
    if (threads < 32) threads = 32;

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto b0_c = b0.contiguous();
    auto b1_c = b1.contiguous();
    auto w_c = weights.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        b0.scalar_type(), "aegis_gate_norm_2branch_fwd", ([&] {
            neuralops::cuda::aegis_gate_norm_2b_kernel<scalar_t><<<total_tokens, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                b0_c.data_ptr<scalar_t>(),
                b1_c.data_ptr<scalar_t>(),
                w_c.data_ptr<float>(),
                total_tokens,
                D,
                eps
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
