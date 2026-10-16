#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void samba_m3_mix_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ m1,
    const scalar_t* __restrict__ m2,
    const scalar_t* __restrict__ la,
    const scalar_t* __restrict__ g3,
    int total_elements, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int token_idx = idx / D;
    float w1 = to_float(g3[token_idx * 3 + 0]);
    float w2 = to_float(g3[token_idx * 3 + 1]);
    float w3 = to_float(g3[token_idx * 3 + 2]);
    float v1 = to_float(m1[idx]);
    float v2 = to_float(m2[idx]);
    float v3 = to_float(la[idx]);
    out[idx] = from_float<scalar_t>(w1 * v1 + w2 * v2 + w3 * v3);
}
}}

torch::Tensor samba_m3_mix_fwd(const torch::Tensor& mamba1, const torch::Tensor& mamba2, const torch::Tensor& linear_attn, const torch::Tensor& gate3) {
    CHECK_INPUT(mamba1); CHECK_INPUT(mamba2); CHECK_INPUT(linear_attn); CHECK_INPUT(gate3);
    auto out = torch::empty_like(mamba1);
    int total = mamba1.numel();
    int D = mamba1.size(-1);
    int threads = 256; int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto m1_c = mamba1.contiguous(); auto m2_c = mamba2.contiguous();
    auto la_c = linear_attn.contiguous(); auto g3_c = gate3.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mamba1.scalar_type(), "samba_m3_mix_fwd", ([&] {
        neuralops::cuda::samba_m3_mix_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), m1_c.data_ptr<scalar_t>(), m2_c.data_ptr<scalar_t>(), la_c.data_ptr<scalar_t>(), g3_c.data_ptr<scalar_t>(),
            total, D
        );
    }));
    return out;
}
