#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Mamba SVD Flash Kernel:
// Factorized low-rank projection with state gating.
// Rather than naive elementwise multiplications, factorizes projection as two sequential steps:
// Step 1: Projects input u (size D) into latent rank space (size R) via V^T: h_r = \sum_d u_d * vt_{r, d}
// Step 2: Projects latent vector h back to dimension D via U: rec_d = \sum_r h_r * u_{d, r}
// Step 3: Fuses decay gate modulation: out_d = rec_d * decay_d
// Thread mapping: 2D Grid over (ceil(N / 128), B) with 128 threads per block.
template <typename scalar_t>
__global__ void __launch_bounds__(128) mamba_svd_flash_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ svd_u,
    const scalar_t* __restrict__ svd_vt,
    const scalar_t* __restrict__ decay,
    const int B,
    const int N,
    const int D,
    const int R
) {
    const int b = blockIdx.y;
    if (b >= B) return;

    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    const int offset = b * N * D + t * D;

    // Allocate thread-local buffer for latent coordinates
    // Rank R is small (e.g., 8, 16, 32, 64)
    float latent[128];
    const int max_r = (R < 128) ? R : 128;

    // Stage 1: Projection to latent space: latent[r] = \sum_d u[d] * svd_vt[r, d]
    for (int r = 0; r < max_r; ++r) {
        float sum = 0.0f;
        const int vt_offset = r * D;
        for (int d = 0; d < D; ++d) {
            sum += to_float(u[offset + d]) * to_float(svd_vt[vt_offset + d]);
        }
        latent[r] = sum;
    }

    // Stage 2: Back-projection to D-space and decay modulation
    for (int d = 0; d < D; ++d) {
        float rec = 0.0f;
        const int u_offset = d * R;
        for (int r = 0; r < max_r; ++r) {
            rec += latent[r] * to_float(svd_u[u_offset + r]);
        }
        const float lam = to_float(decay[offset + d]);
        out[offset + d] = from_float<scalar_t>(rec * lam);
    }
}

}} // namespace neuralops::cuda

torch::Tensor mamba_svd_flash_fwd(
    const torch::Tensor& u,
    const torch::Tensor& svd_u,
    const torch::Tensor& svd_vt,
    const torch::Tensor& decay
) {
    CHECK_INPUT(u);
    CHECK_INPUT(svd_u);
    CHECK_INPUT(svd_vt);
    CHECK_INPUT(decay);

    const int B = u.size(0);
    const int N = u.size(1);
    const int D = u.size(2);
    const int R = svd_u.size(1);

    auto out = torch::empty_like(u);
    if (B == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (N + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto u_c = u.contiguous();
    auto u_mat = svd_u.contiguous();
    auto vt_mat = svd_vt.contiguous();
    auto d_c = decay.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        u.scalar_type(), "mamba_svd_flash_fwd", ([&] {
            neuralops::cuda::mamba_svd_flash_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                u_c.data_ptr<scalar_t>(),
                u_mat.data_ptr<scalar_t>(),
                vt_mat.data_ptr<scalar_t>(),
                d_c.data_ptr<scalar_t>(),
                B, N, D, R
            );
        })
    );

    return out;
}
