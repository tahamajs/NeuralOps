// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(256) spectral_shrinkage_vectorized_kernel(
    scalar_t* __restrict__ y_re,
    scalar_t* __restrict__ y_im,
    const scalar_t* __restrict__ x_re,
    const scalar_t* __restrict__ x_im,
    const scalar_t* __restrict__ w_re,
    const scalar_t* __restrict__ w_im,
    const scalar_t* __restrict__ tau,
    const int total_elements
) {
    const int num_vec = total_elements / 4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < num_vec; idx += stride) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 xr4 = *reinterpret_cast<const float4*>(x_re + idx * 4);
            const float4 xi4 = *reinterpret_cast<const float4*>(x_im + idx * 4);
            const float4 wr4 = *reinterpret_cast<const float4*>(w_re + idx * 4);
            const float4 wi4 = *reinterpret_cast<const float4*>(w_im + idx * 4);
            const float4 tau4 = *reinterpret_cast<const float4*>(tau + idx * 4);

            float4 yr4;
            float4 yi4;

            #define COMPUTE_SHRINK(xr, xi, wr, wi, tau_v, yr_out, yi_out) { \
                const float yr_prod = (xr) * (wr) - (xi) * (wi); \
                const float yi_prod = (xr) * (wi) + (xi) * (wr); \
                const float mag = sqrtf(yr_prod * yr_prod + yi_prod * yi_prod + 1e-12f); \
                const float sig_tau = 1.0f / (1.0f + expf(-tau_v)); \
                const float scale = fmaxf(0.0f, (mag - sig_tau) / mag); \
                yr_out = yr_prod * scale; \
                yi_out = yi_prod * scale; \
            }
            COMPUTE_SHRINK(xr4.x, xi4.x, wr4.x, wi4.x, tau4.x, yr4.x, yi4.x);
            COMPUTE_SHRINK(xr4.y, xi4.y, wr4.y, wi4.y, tau4.y, yr4.y, yi4.y);
            COMPUTE_SHRINK(xr4.z, xi4.z, wr4.z, wi4.z, tau4.z, yr4.z, yi4.z);
            COMPUTE_SHRINK(xr4.w, xi4.w, wr4.w, wi4.w, tau4.w, yr4.w, yi4.w);
            #undef COMPUTE_SHRINK

            *reinterpret_cast<float4*>(y_re + idx * 4) = yr4;
            *reinterpret_cast<float4*>(y_im + idx * 4) = yi4;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int elem_idx = idx * 4 + k;
                const float xr = to_float(x_re[elem_idx]);
                const float xi = to_float(x_im[elem_idx]);
                const float wr = to_float(w_re[elem_idx]);
                const float wi = to_float(w_im[elem_idx]);

                const float yr = xr * wr - xi * wi;
                const float yi = xr * wi + xi * wr;

                const float mag = sqrtf(yr * yr + yi * yi + 1e-12f);
                const float sig_tau = 1.0f / (1.0f + expf(-to_float(tau[elem_idx])));
                const float scale = fmaxf(0.0f, (mag - sig_tau) / mag);

                y_re[elem_idx] = from_float<scalar_t>(yr * scale);
                y_im[elem_idx] = from_float<scalar_t>(yi * scale);
            }
        }
    }

    const int rem_start = num_vec * 4;
    for (int idx = rem_start + tid; idx < total_elements; idx += stride) {
        const float xr = to_float(x_re[idx]);
        const float xi = to_float(x_im[idx]);
        const float wr = to_float(w_re[idx]);
        const float wi = to_float(w_im[idx]);

        const float yr = xr * wr - xi * wi;
        const float yi = xr * wi + xi * wr;

        const float mag = sqrtf(yr * yr + yi * yi + 1e-12f);
        const float sig_tau = 1.0f / (1.0f + expf(-to_float(tau[idx])));
        const float scale = fmaxf(0.0f, (mag - sig_tau) / mag);

        y_re[idx] = from_float<scalar_t>(yr * scale);
        y_im[idx] = from_float<scalar_t>(yi * scale);
    }
}

}}

std::vector<torch::Tensor> spectral_shrinkage_fwd(
    const torch::Tensor& x_re,
    const torch::Tensor& x_im,
    const torch::Tensor& w_re,
    const torch::Tensor& w_im,
    const torch::Tensor& tau
) {
    CHECK_INPUT(x_re);
    CHECK_INPUT(x_im);
    CHECK_INPUT(w_re);
    CHECK_INPUT(w_im);
    CHECK_INPUT(tau);

    TORCH_CHECK(x_im.sizes() == x_re.sizes(), "x_im shape mismatch");
    TORCH_CHECK(w_re.sizes() == x_re.sizes(), "w_re shape mismatch");
    TORCH_CHECK(w_im.sizes() == x_re.sizes(), "w_im shape mismatch");
    TORCH_CHECK(tau.sizes() == x_re.sizes(), "tau shape mismatch");

    auto y_re = torch::empty_like(x_re);
    auto y_im = torch::empty_like(x_im);
    const int total = x_re.numel();
    if (total == 0) return {y_re, y_im};

    const int vec_count = (total + 3) / 4;
    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (vec_count + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto xr_c = x_re.contiguous();
    auto xi_c = x_im.contiguous();
    auto wr_c = w_re.contiguous();
    auto wi_c = w_im.contiguous();
    auto tau_c = tau.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x_re.scalar_type(), "spectral_shrinkage_fwd", ([&] {
            neuralops::cuda::spectral_shrinkage_vectorized_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                y_re.data_ptr<scalar_t>(),
                y_im.data_ptr<scalar_t>(),
                xr_c.data_ptr<scalar_t>(),
                xi_c.data_ptr<scalar_t>(),
                wr_c.data_ptr<scalar_t>(),
                wi_c.data_ptr<scalar_t>(),
                tau_c.data_ptr<scalar_t>(),
                total
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {y_re, y_im};
}
