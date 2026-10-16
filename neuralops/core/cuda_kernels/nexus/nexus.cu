#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Nexus Non-Linear Attention Kernel:
// Projects features via tanh non-linear mapping and causal recurrence:
// \phi(k_t) = \tanh(\tau \cdot k_t)
// s_t = s_{t-1} + \phi(k_t) \cdot v_t
// y_t = \tanh(\tau \cdot q_t) \cdot s_t
// Thread mapping: 2D Grid over (ceil(D / 128), B * H) with 128 threads per block.
// Eliminates the hardcoded threads = D, blocks = B * H anti-pattern.
template <typename scalar_t>
__global__ void __launch_bounds__(128) nexus_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int N,
    const int D,
    const float tau
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    const int head_offset = bh * (N * D);
    float s = 0.0f;

    for (int t = 0; t < N; ++t) {
        const int idx = head_offset + t * D + d;
        const float k_val = to_float(k[idx]);
        const float v_val = to_float(v[idx]);
        const float phi_k = tanhf(k_val * tau);
        s += phi_k * v_val;

        const float q_val = to_float(q[idx]);
        const float phi_q = tanhf(q_val * tau);
        out[idx] = from_float<scalar_t>(phi_q * s);
    }
}

}} // namespace neuralops::cuda

torch::Tensor nexus_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    double tau
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "nexus_fwd", ([&] {
            neuralops::cuda::nexus_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, N, D, static_cast<float>(tau)
            );
        })
    );

    return out;
}
