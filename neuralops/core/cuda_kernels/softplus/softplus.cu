#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// 128-bit Vectorized Softplus Forward Kernel:
// y = (1 / beta) * log1p(exp(beta * x)) if beta * x <= threshold else x
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

    const int rem_start = num_vec * 4;
    for (int idx = rem_start + tid; idx < total_elements; idx += stride) {
        const float x_val = to_float(x[idx]);
        const float bx = beta * x_val;
        const float y_val = (bx > threshold) ? x_val : (inv_beta * log1pf(expf(bx)));
        out[idx] = from_float<scalar_t>(y_val);
    }
}

// 128-bit Vectorized Softplus Backward Kernel:
// dx = dout * sigmoid(beta * x) = dout / (1 + exp(-beta * x)) if beta * x <= threshold else dout
template <typename scalar_t>
__launch_bounds__(256)
__global__ void softplus_bwd_vectorized_kernel(
    scalar_t* __restrict__ dx,
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ x,
    const int total_elements,
    const float beta,
    const float threshold
) {
    const int num_vec = total_elements / 4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < num_vec; idx += stride) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 dout_v = *reinterpret_cast<const float4*>(dout + idx * 4);
            const float4 x_v = *reinterpret_cast<const float4*>(x + idx * 4);
            float4 dx_v;

            #define EVAL_SOFTPLUS_BWD(d, val) { \
                const float bx = beta * (val); \
                (d) = (bx > threshold) ? (d) : ((d) / (1.0f + expf(-bx))); \
            }
            dx_v.x = dout_v.x; EVAL_SOFTPLUS_BWD(dx_v.x, x_v.x);
            dx_v.y = dout_v.y; EVAL_SOFTPLUS_BWD(dx_v.y, x_v.y);
            dx_v.z = dout_v.z; EVAL_SOFTPLUS_BWD(dx_v.z, x_v.z);
            dx_v.w = dout_v.w; EVAL_SOFTPLUS_BWD(dx_v.w, x_v.w);
            #undef EVAL_SOFTPLUS_BWD

            *reinterpret_cast<float4*>(dx + idx * 4) = dx_v;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int elem_idx = idx * 4 + k;
                const float x_val = to_float(x[elem_idx]);
                const float d_val = to_float(dout[elem_idx]);
                const float bx = beta * x_val;
                const float g_val = (bx > threshold) ? d_val : (d_val / (1.0f + expf(-bx)));
                dx[elem_idx] = from_float<scalar_t>(g_val);
            }
        }
    }

    const int rem_start = num_vec * 4;
    for (int idx = rem_start + tid; idx < total_elements; idx += stride) {
        const float x_val = to_float(x[idx]);
        const float d_val = to_float(dout[idx]);
        const float bx = beta * x_val;
        const float g_val = (bx > threshold) ? d_val : (d_val / (1.0f + expf(-bx)));
        dx[idx] = from_float<scalar_t>(g_val);
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

torch::Tensor softplus_bwd(
    const torch::Tensor& dout,
    const torch::Tensor& x,
    const float beta,
    const float threshold
) {
    TORCH_CHECK(dout.is_cuda(), "dout must be a CUDA tensor");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");

    auto dx = torch::empty_like(x);
    const int total_elements = x.numel();
    if (total_elements == 0) return dx;

    const int vec_count = (total_elements + 3) / 4;
    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (vec_count + THREADS - 1) / THREADS);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto dout_c = dout.contiguous();
    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "softplus_bwd", ([&] {
            neuralops::cuda::softplus_bwd_vectorized_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                dx.data_ptr<scalar_t>(),
                dout_c.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                total_elements,
                beta,
                threshold
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return dx;
}
