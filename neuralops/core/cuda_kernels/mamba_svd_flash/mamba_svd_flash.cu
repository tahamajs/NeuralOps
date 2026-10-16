#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void mamba_svd_flash_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ svd_u,
    const scalar_t* __restrict__ svd_vt,
    const scalar_t* __restrict__ decay,
    int B, int N, int D, int R
) {
    int b = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        float lam = to_float(decay[b * N * D + t * D + d]);
        float u_val = to_float(u[b * N * D + t * D + d]);
        for (int r = 0; r < R; ++r) {
            float u_proj = to_float(svd_u[d * R + r]);
            float vt_proj = to_float(svd_vt[r * D + d]);
            acc += u_proj * vt_proj * u_val;
        }
        out[b * N * D + t * D + d] = from_float<scalar_t>(lam * acc);
    }
}
}}

torch::Tensor mamba_svd_flash_fwd(const torch::Tensor& u, const torch::Tensor& svd_u, const torch::Tensor& svd_vt, const torch::Tensor& decay) {
    CHECK_INPUT(u); CHECK_INPUT(svd_u); CHECK_INPUT(svd_vt); CHECK_INPUT(decay);
    int B = u.size(0); int N = u.size(1); int D = u.size(2); int R = svd_u.size(1);
    auto out = torch::empty_like(u);
    int threads = 128; int blocks_x = (N + threads - 1) / threads;
    dim3 grid(blocks_x, B);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto u_c = u.contiguous(); auto u_mat = svd_u.contiguous(); auto vt_mat = svd_vt.contiguous(); auto d_c = decay.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, u.scalar_type(), "mamba_svd_flash_fwd", ([&] {
        neuralops::cuda::mamba_svd_flash_fwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), u_c.data_ptr<scalar_t>(), u_mat.data_ptr<scalar_t>(), vt_mat.data_ptr<scalar_t>(), d_c.data_ptr<scalar_t>(),
            B, N, D, R
        );
    }));
    return out;
}
