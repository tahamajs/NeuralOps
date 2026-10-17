#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// 128-bit Vectorized Softplus Kernel:
// Form: y = (1 / beta) * log(1 + exp(beta * x)) if beta * x <= threshold else x
// Employs float4 128-bit vectorized global memory load/store for maximum memory bandwidth.
template <typename scalar_t>
__launch_bounds__(256)
__global__ void softplus_fwd_vectorized_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const int total_elements,
    const float beta,
    const float inv_beta,
    const float threshold
) {
    const int num_vec = total_elements / 4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < num_vec; idx += stride) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 in_v = *reinterpret_cast<const float4*>(x + idx * 4);
            float4 out_v;

            #define EVAL_SOFTPLUS(val) { \
                const float bx = beta * (val); \
                (val) = (bx > threshold) ? (val) : (inv_beta * log1pf(expf(bx))); \
            }
            out_v.x = in_v.x; EVAL_SOFTPLUS(out_v.x);
            out_v.y = in_v.y; EVAL_SOFTPLUS(out_v.y);
            out_v.z = in_v.z; EVAL_SOFTPLUS(out_v.z);
            out_v.w = in_v.w; EVAL_SOFTPLUS(out_v.w);
            #undef EVAL_SOFTPLUS

            *reinterpret_cast<float4*>(out + idx * 4) = out_v;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int elem_idx = idx * 4 + k;
                const float x_val = to_float(x[elem_idx]);
                const float bx = beta * x_val;
                const float y_val = (bx > threshold) ? x_val : (inv_beta * log1pf(expf(bx)));
                out[elem_idx] = from_float<scalar_t>(y_val);
            }
        }
    }

    // Remainder loop for trailing unaligned elements
    const int rem_start = num_vec * 4;
    for (int idx = rem_start + tid; idx < total_elements; idx += stride) {
        const float x_val = to_float(x[idx]);
        const float bx = beta * x_val;
        const float y_val = (bx > threshold) ? x_val : (inv_beta * log1pf(expf(bx)));
        out[idx] = from_float<scalar_t>(y_val);
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor softplus_fwd(
    const torch::Tensor& x,
    const float beta,
    const float threshold
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");

    auto out = torch::empty_like(x);
    const int total_elements = x.numel();
    if (total_elements == 0) return out;

    const int vec_count = (total_elements + 3) / 4;
    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (vec_count + THREADS - 1) / THREADS);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto x_c = x.contiguous();
    const float inv_beta = 1.0f / beta;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "softplus_fwd", ([&] {
            neuralops::cuda::softplus_fwd_vectorized_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                total_elements,
                beta,
                inv_beta,
                threshold
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
