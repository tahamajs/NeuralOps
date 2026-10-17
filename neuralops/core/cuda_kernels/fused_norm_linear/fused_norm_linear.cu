#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

// Fused LayerNorm + Linear Forward Kernel
// x: [M, K], weight: [N, K], bias: [N] (optional) -> out: [M, N]
template <typename scalar_t>
__launch_bounds__(128)
__global__ void fused_norm_linear_fwd_kernel(
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

    // Step 1: Compute mean across K
    float sum_x = 0.0f;
    for (int k = tid; k < K; k += num_threads) {
        sum_x += to_float(x[m * K + k]);
    }
    sum_x = block_reduce_sum(sum_x);
    __shared__ float s_mean;
    if (tid == 0) {
        s_mean = sum_x / (float)K;
    }
    __syncthreads();
    const float mean = s_mean;

    // Step 2: Compute variance across K
    float sum_sq = 0.0f;
    for (int k = tid; k < K; k += num_threads) {
        float diff = to_float(x[m * K + k]) - mean;
        sum_sq += diff * diff;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_rstd;
    if (tid == 0) {
        s_rstd = rsqrtf(sum_sq / (float)K + eps);
    }
    __syncthreads();
    const float rstd = s_rstd;

    extern __shared__ float s_x_hat[];

    // Step 3: Compute x_hat and store in shared memory (and global memory if requested)
    for (int k = tid; k < K; k += num_threads) {
        float x_norm = (to_float(x[m * K + k]) - mean) * rstd;
        s_x_hat[k] = x_norm;
        if (x_hat_out != nullptr) {
            x_hat_out[m * K + k] = from_float<scalar_t>(x_norm);
        }
    }
    __syncthreads();

    // Step 4: Compute out[m, n] = x_hat @ weight.T + bias using cached shared memory
    for (int n = tid; n < N; n += num_threads) {
        float y = 0.0f;
        for (int k = 0; k < K; ++k) {
            y += s_x_hat[k] * to_float(weight[n * K + k]);
        }
        if (has_bias && bias != nullptr) {
            y += to_float(bias[n]);
        }
        out[m * N + n] = from_float<scalar_t>(y);
    }
}

// Backward Kernel for dX: dX = LayerNormBackward(dout @ weight)
template <typename scalar_t>
__launch_bounds__(128)
__global__ void fused_norm_linear_bwd_dx_kernel(
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

    // Mean and rstd of x[m, :]
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
        float diff = to_float(x[m * K + k]) - mean;
        sum_sq += diff * diff;
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rsqrtf(sum_sq / (float)K + eps);
    __syncthreads();
    const float rstd = s_rstd;

    // dx_hat = dout[m, :] @ weight
    // Compute mean(dx_hat) and mean(dx_hat * x_hat)
    float sum_dx_hat = 0.0f;
    float sum_dx_hat_x_hat = 0.0f;

    for (int k = tid; k < K; k += num_threads) {
        float dx_h = 0.0f;
        for (int n = 0; n < N; ++n) {
            dx_h += to_float(dout[m * N + n]) * to_float(weight[n * K + k]);
        }
        float x_hat = (to_float(x[m * K + k]) - mean) * rstd;
        sum_dx_hat += dx_h;
        sum_dx_hat_x_hat += dx_h * x_hat;
    }

    sum_dx_hat = block_reduce_sum(sum_dx_hat);
    sum_dx_hat_x_hat = block_reduce_sum(sum_dx_hat_x_hat);

    __shared__ float s_mean_dx_hat;
    __shared__ float s_mean_dx_hat_x_hat;
    if (tid == 0) {
        s_mean_dx_hat = sum_dx_hat / (float)K;
        s_mean_dx_hat_x_hat = sum_dx_hat_x_hat / (float)K;
    }
    __syncthreads();

    const float mean_dx_hat = s_mean_dx_hat;
    const float mean_dx_hat_x_hat = s_mean_dx_hat_x_hat;

    for (int k = tid; k < K; k += num_threads) {
        float dx_h = 0.0f;
        for (int n = 0; n < N; ++n) {
            dx_h += to_float(dout[m * N + n]) * to_float(weight[n * K + k]);
        }
        float x_hat = (to_float(x[m * K + k]) - mean) * rstd;
        float dx_val = rstd * (dx_h - mean_dx_hat - x_hat * mean_dx_hat_x_hat);
        dx[m * K + k] = from_float<scalar_t>(dx_val);
    }
}

} // namespace cuda
} // namespace neuralops

std::vector<torch::Tensor> fused_norm_linear_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const float eps
) {
    CHECK_INPUT(x);
    CHECK_INPUT(weight);

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);
    const bool has_bias = bias.defined() && bias.numel() > 0;

    auto out = torch::empty({M, N}, x.options());
    auto x_hat = torch::empty_like(x);
    if (M == 0 || K == 0 || N == 0) return {out, x_hat};

    const int threads = 128;
    const size_t shmem_size = K * sizeof(float);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fused_norm_linear_fwd", ([&] {
            neuralops::cuda::fused_norm_linear_fwd_kernel<scalar_t><<<M, threads, shmem_size, stream>>>(
                out.data_ptr<scalar_t>(), x_hat.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(), has_bias ? bias.data_ptr<scalar_t>() : nullptr,
                M, K, N, eps, has_bias
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {out, x_hat};
}

torch::Tensor fused_norm_linear_bwd_dx(
    const torch::Tensor& dout,
    const torch::Tensor& weight,
    const torch::Tensor& x,
    const float eps
) {
    CHECK_INPUT(dout);
    CHECK_INPUT(weight);
    CHECK_INPUT(x);

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = weight.size(0);

    auto dx = torch::empty_like(x);
    const int threads = 128;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fused_norm_linear_bwd_dx", ([&] {
            neuralops::cuda::fused_norm_linear_bwd_dx_kernel<scalar_t><<<M, threads, 0, stream>>>(
                dx.data_ptr<scalar_t>(), dout.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(), M, K, N, eps
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return dx;
}
