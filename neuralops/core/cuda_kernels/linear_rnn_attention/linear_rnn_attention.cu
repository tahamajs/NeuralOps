#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Linear RNN temporal recurrence with decoupled 2D block grid
template <typename scalar_t>
__global__ void __launch_bounds__(256) linear_rnn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b_mat,
    const int B,
    const int N,
    const int D
) {
    const int b_idx = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (b_idx >= B || d >= D) return;

    float h = 0.0f;
    for (int t = 0; t < N; ++t) {
        const int idx = b_idx * N * D + t * D + d;
        const float a_val = to_float(a[idx]);
        const float b_val = to_float(b_mat[idx]);
        const float x_val = to_float(x[idx]);

        h = a_val * h + b_val * x_val;
        out[idx] = from_float<scalar_t>(h);
    }
}

}} // namespace neuralops::cuda

torch::Tensor linear_rnn_fwd(
    const torch::Tensor& x,
    const torch::Tensor& a,
    const torch::Tensor& b_mat
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b_mat);

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    auto out = torch::empty_like(x);

    if (B == 0 || D == 0) return out;

    constexpr int THREADS = 256;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto a_c = a.contiguous();
    auto b_c = b_mat.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "linear_rnn_fwd", ([&] {
            neuralops::cuda::linear_rnn_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                a_c.data_ptr<scalar_t>(),
                b_c.data_ptr<scalar_t>(),
                B, N, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
