#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void jamba_mix_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ mamba,
    const scalar_t* __restrict__ attn,
    const scalar_t* __restrict__ gate,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float m_val = to_float(mamba[idx]);
    float a_val = to_float(attn[idx]);
    float g_val = 1.0f / (1.0f + expf(-to_float(gate[idx])));
    out[idx] = from_float<scalar_t>(g_val * m_val + (1.0f - g_val) * a_val);
}
}}

torch::Tensor jamba_mix_fwd(const torch::Tensor& mamba_out, const torch::Tensor& attn_out, const torch::Tensor& gate) {
    CHECK_INPUT(mamba_out); CHECK_INPUT(attn_out); CHECK_INPUT(gate);
    auto out = torch::empty_like(mamba_out);
    int total = mamba_out.numel();
    int threads = 256; int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto m_c = mamba_out.contiguous(); auto a_c = attn_out.contiguous(); auto g_c = gate.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mamba_out.scalar_type(), "jamba_mix_fwd", ([&] {
        neuralops::cuda::jamba_mix_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), m_c.data_ptr<scalar_t>(), a_c.data_ptr<scalar_t>(), g_c.data_ptr<scalar_t>(), total
        );
    }));
    return out;
}
