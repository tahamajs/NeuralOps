#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Unnormalized Gated Linear Attention state accumulation without denominator division
template <typename scalar_t>
__global__ void __launch_bounds__(256) gla_unnorm_scan_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ g,
    const int B,
    const int H,
    const int N,
    const int D
) {
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || h >= H || d >= D) return;

    float s = 0.0f;
    for (int t = 0; t < N; ++t) {
        const int idx = (b * H + h) * N * D + t * D + d;
        const float gate = expf(to_float(g[idx]));
        const float k_val = to_float(k[idx]);
        const float v_val = to_float(v[idx]);

        s = gate * s + k_val * v_val;
        const float q_val = to_float(q[idx]);
        out[idx] = from_float<scalar_t>(q_val * s);
    }
}

}} // namespace neuralops::cuda

torch::Tensor gla_unnormalized_scan_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(g);

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    auto out = torch::empty_like(q);

    if (B == 0 || H == 0 || D == 0) return out;

    constexpr int THREADS = 256;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto g_c = g.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "gla_unnormalized_scan_fwd", ([&] {
            neuralops::cuda::gla_unnorm_scan_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                B, H, N, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
