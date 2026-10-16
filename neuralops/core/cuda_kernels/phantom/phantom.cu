#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void phantom_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gate,
    int total_elements, float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float x_val = to_float(x[idx]);
    float g_val = 1.0f / (1.0f + expf(-to_float(gate[idx])));
    float mask = (g_val >= threshold) ? g_val : 0.0f;
    out[idx] = from_float<scalar_t>(x_val * mask);
}
}}

torch::Tensor phantom_fwd(const torch::Tensor& x, const torch::Tensor& gate, double threshold) {
    CHECK_INPUT(x); CHECK_INPUT(gate);
    auto out = torch::empty_like(x);
    int total = x.numel();
    int threads = 256; int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous(); auto g_c = gate.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "phantom_fwd", ([&] {
        neuralops::cuda::phantom_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), g_c.data_ptr<scalar_t>(), total, (float)threshold
        );
    }));
    return out;
}
