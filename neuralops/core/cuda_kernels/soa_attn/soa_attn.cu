#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Second-order attention polynomial feature expansion kernel
template <typename scalar_t>
__global__ void __launch_bounds__(256) soa_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int total_elements,
    const float alpha
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        const float q_val = to_float(q[idx]);
        const float k_val = to_float(k[idx]);
        const float v_val = to_float(v[idx]);
        const float qk = q_val * k_val;
        const float phi = qk + alpha * (qk * qk);
        out[idx] = from_float<scalar_t>(phi * v_val);
    }
}

}} // namespace neuralops::cuda

torch::Tensor soa_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    double alpha
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    auto out = torch::empty_like(q);
    const int total = q.numel();
    if (total == 0) return out;

    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (total + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "soa_attn_fwd", ([&] {
            neuralops::cuda::soa_attn_fwd_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                total,
                static_cast<float>(alpha)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
