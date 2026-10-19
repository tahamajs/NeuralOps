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

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__launch_bounds__(128)
__global__ void fused_norm_linear_fwd_vectorized_kernel(
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ x_hat_out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int M,
    const int K,
    const int N,
    const float eps,
    const bool has_bias
) {
    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int num_vec_K = K / 4;
    const int rem_start_K = num_vec_K * 4;
    const int row_base = m * K;

    float sum_x = 0.0f;
    for (int vk = tid; vk < num_vec_K; vk += num_threads) {
        const int k = vk * 4;
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 in4 = *reinterpret_cast<const float4*>(x + row_base + k);
            sum_x += in4.x + in4.y + in4.z + in4.w;
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                sum_x += to_float(x[row_base + k + j]);
            }
        }
    }
    for (int k = rem_start_K + tid; k < K; k += num_threads) {
        sum_x += to_float(x[row_base + k]);
    }
    sum_x = block_reduce_sum(sum_x);
    __shared__ float s_mean;
    if (tid == 0) {
        s_mean = sum_x / (float)K;
    }
    __syncthreads();
    const float mean = s_mean;

    float sum_sq = 0.0f;
    for (int vk = tid; vk < num_vec_K; vk += num_threads) {
        const int k = vk * 4;
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 in4 = *reinterpret_cast<const float4*>(x + row_base + k);
            const float d0 = in4.x - mean;
            const float d1 = in4.y - mean;
            const float d2 = in4.z - mean;
            const float d3 = in4.w - mean;
            sum_sq = fmaf(d0, d0, sum_sq);
            sum_sq = fmaf(d1, d1, sum_sq);
            sum_sq = fmaf(d2, d2, sum_sq);
            sum_sq = fmaf(d3, d3, sum_sq);
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float diff = to_float(x[row_base + k + j]) - mean;
                sum_sq = fmaf(diff, diff, sum_sq);
            }
        }
    }
    for (int k = rem_start_K + tid; k < K; k += num_threads) {
        const float diff = to_float(x[row_base + k]) - mean;
        sum_sq = fmaf(diff, diff, sum_sq);
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_rstd;
    if (tid == 0) {
        s_rstd = rsqrtf(sum_sq / (float)K + eps);
    }
    __syncthreads();
    const float rstd = s_rstd;

    extern __shared__ float s_x_hat[];

    for (int k = tid; k < K; k += num_threads) {
        const float x_norm = (to_float(x[row_base + k]) - mean) * rstd;
        s_x_hat[k] = x_norm;
        if (x_hat_out != nullptr) {
            x_hat_out[row_base + k] = from_float<scalar_t>(x_norm);
        }
    }
    __syncthreads();

    const int num_vec_K_inner = K / 4;
    const int rem_start_K_inner = num_vec_K_inner * 4;

    for (int n = tid; n < N; n += num_threads) {
        float y = (has_bias && bias != nullptr) ? to_float(bias[n]) : 0.0f;
        const int w_row = n * K;

        for (int vk = 0; vk < num_vec_K_inner; ++vk) {
            const int k = vk * 4;
            if constexpr (std::is_same_v<scalar_t, float>) {
                const float4 w4 = *reinterpret_cast<const float4*>(weight + w_row + k);
                y = fmaf(s_x_hat[k], w4.x, y);
                y = fmaf(s_x_hat[k + 1], w4.y, y);
                y = fmaf(s_x_hat[k + 2], w4.z, y);
                y = fmaf(s_x_hat[k + 3], w4.w, y);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    y = fmaf(s_x_hat[k + j], to_float(weight[w_row + k + j]), y);
                }
            }
        }
        for (int k = rem_start_K_inner; k < K; ++k) {
            y = fmaf(s_x_hat[k], to_float(weight[w_row + k]), y);
        }
        out[m * N + n] = from_float<scalar_t>(y);
    }
}

template <typename scalar_t>
__launch_bounds__(128)
__global__ void fused_norm_linear_bwd_dx_vectorized_kernel(
    scalar_t* __restrict__ dx,
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ x,
    const int M,
    const int K,
    const int N,
    const float eps
) {
    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    float sum_x = 0.0f;
    for (int k = tid; k < K; k += num_threads) {
        sum_x += to_float(x[m * K + k]);
    }
    sum_x = block_reduce_sum(sum_x);
    __shared__ float s_mean;
    if (tid == 0) s_mean = sum_x / (float)K;
    __syncthreads();
    const float mean = s_mean;

    float sum_sq = 0.0f;
    for (int k = tid; k < K; k += num_threads) {
        const float diff = to_float(x[m * K + k]) - mean;
        sum_sq = fmaf(diff, diff, sum_sq);
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rsqrtf(sum_sq / (float)K + eps);
    __syncthreads();
    const float rstd = s_rstd;

    extern __shared__ float s_dy[];

    for (int k = tid; k < K; k += num_threads) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            acc = fmaf(to_float(dout[m * N + n]), to_float(weight[n * K + k]), acc);
        }
        s_dy[k] = acc;
    }
    __syncthreads();

    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (int k = tid; k < K; k += num_threads) {
        const float dy_val = s_dy[k];
        const float x_hat = (to_float(x[m * K + k]) - mean) * rstd;
        sum_dy += dy_val;
        sum_dy_xhat = fmaf(dy_val, x_hat, sum_dy_xhat);
    }
    sum_dy = block_reduce_sum(sum_dy);
    sum_dy_xhat = block_reduce_sum(sum_dy_xhat);

    __shared__ float s_sum_dy, s_sum_dy_xhat;
    if (tid == 0) {
        s_sum_dy = sum_dy;
        s_sum_dy_xhat = sum_dy_xhat;
    }
    __syncthreads();

    const float mean_dy = s_sum_dy / (float)K;
    const float mean_dy_xhat = s_sum_dy_xhat / (float)K;

    for (int k = tid; k < K; k += num_threads) {
        const float x_hat = (to_float(x[m * K + k]) - mean) * rstd;
        const float dx_val = rstd * (s_dy[k] - mean_dy - x_hat * mean_dy_xhat);
        dx[m * K + k] = from_float<scalar_t>(dx_val);
    }
}

}
}

std::vector<torch::Tensor> fused_norm_linear_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    float eps = 1e-5f
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();
    const int M = x_2d.size(0);
    const int K = x_2d.size(1);
    const int N = weight.size(0);

    TORCH_CHECK(weight.size(1) == K, "weight second dim must match x last dim");

    const bool has_bias = bias.defined() && bias.numel() > 0;
    if (has_bias) CHECK_INPUT(bias);

    auto out = torch::empty({M, N}, x.options());
    auto x_hat = torch::empty_like(x_2d);

    if (M == 0 || K == 0 || N == 0) {
        std::vector<int64_t> out_shape(orig_shape.begin(), orig_shape.end() - 1);
        out_shape.push_back(N);
        return {out.reshape(out_shape), x_hat.reshape(orig_shape)};
    }

    const int threads = 128;
    const size_t shared_mem = K * sizeof(float);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto weight_c = weight.contiguous();
    auto bias_c = has_bias ? bias.contiguous() : torch::empty(0, x.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fused_norm_linear_fwd", ([&] {
            neuralops::cuda::fused_norm_linear_fwd_vectorized_kernel<scalar_t><<<M, threads, shared_mem, stream>>>(
                out.data_ptr<scalar_t>(),
                x_hat.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                weight_c.data_ptr<scalar_t>(),
                has_bias ? bias_c.data_ptr<scalar_t>() : nullptr,
                M, K, N, eps, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();

    std::vector<int64_t> out_final_shape(orig_shape.begin(), orig_shape.end() - 1);
    out_final_shape.push_back(N);
    return {out.reshape(out_final_shape), x_hat.reshape(orig_shape)};
}

torch::Tensor fused_norm_linear_bwd_dx(
    const torch::Tensor& dout,
    const torch::Tensor& weight,
    const torch::Tensor& x,
    const float eps
) {
    CHECK_INPUT(dout);
    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    auto orig_shape = x.sizes();
    auto dout_2d = dout.reshape({-1, dout.size(-1)}).contiguous();
    auto x_2d = x.reshape({-1, orig_shape.back()}).contiguous();

    const int M = x_2d.size(0);
    const int K = x_2d.size(1);
    const int N = weight.size(0);

    auto dx = torch::empty_like(x_2d);

    const int threads = 128;
    const size_t shared_mem = K * sizeof(float);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto weight_c = weight.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fused_norm_linear_bwd_dx", ([&] {
            neuralops::cuda::fused_norm_linear_bwd_dx_vectorized_kernel<scalar_t><<<M, threads, shared_mem, stream>>>(
                dx.data_ptr<scalar_t>(),
                dout_2d.data_ptr<scalar_t>(),
                weight_c.data_ptr<scalar_t>(),
                x_2d.data_ptr<scalar_t>(),
                M, K, N, eps
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return dx.reshape(orig_shape);
}

