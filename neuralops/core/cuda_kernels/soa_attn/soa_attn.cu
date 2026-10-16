#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void soa_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    int total_elements, float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float q_val = to_float(q[idx]);
    float k_val = to_float(k[idx]);
    float v_val = to_float(v[idx]);
    // Second-order attention polynomial feature mapping
    float phi = (q_val * k_val) + alpha * (q_val * q_val * k_val * k_val);
    out[idx] = from_float<scalar_t>(phi * v_val);
}
}}

torch::Tensor soa_attn_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, double alpha) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);
    auto out = torch::empty_like(q);
    int total = q.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous(); auto k_c = k.contiguous(); auto v_c = v.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "soa_attn_fwd", ([&] {
        neuralops::cuda::soa_attn_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(), total, (float)alpha
        );
    }));
    return out;
}
