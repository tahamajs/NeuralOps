#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Highly optimized streaming CUDA kernel for factorized Mamba SVD projection
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
    const int R,
    const bool decay_has_batch
) {
    const int b = blockIdx.y;
    const int t = blockIdx.x;
    if (b >= B || t >= N) return;

    const int tid = threadIdx.x;
    const int base_offset = (b * N + t) * D;

    extern __shared__ float smem[];
    float* s_u = smem;               // size D
    float* s_latent = smem + D;       // size R

    // Step 1: Collaborative coalesced load of u[b, t, :] into shared memory
    for (int d = tid; d < D; d += blockDim.x) {
        s_u[d] = to_float(u[base_offset + d]);
    }
    __syncthreads();

    // Step 2: Thread-parallel projection into latent coordinates h_r = \sum_d u_d * vt_{r, d}
    for (int r = tid; r < R; r += blockDim.x) {
        float sum = 0.0f;
        const int vt_offset = r * D;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            sum = fmaf(s_u[d], to_float(svd_vt[vt_offset + d]), sum);
        }
        s_latent[r] = sum;
    }
    __syncthreads();

    // Step 3: Reconstruction out_d = (\sum_r h_r * u_{d, r}) * decay_d
    for (int d = tid; d < D; d += blockDim.x) {
        float rec = 0.0f;
        const int u_mat_offset = d * R;
        #pragma unroll 4
        for (int r = 0; r < R; ++r) {
            rec = fmaf(s_latent[r], to_float(svd_u[u_mat_offset + r]), rec);
        }
        const float lam = to_float(decay_has_batch ? decay[base_offset + d] : decay[d]);
        out[base_offset + d] = from_float<scalar_t>(rec * lam);
    }
}

}} // namespace neuralops::cuda

torch::Tensor mamba_svd_flash_fwd(
    const torch::Tensor& u,
    const torch::Tensor& svd_u,
    const torch::Tensor& svd_vt,
    const torch::Tensor& decay
) {
    TORCH_CHECK(u.is_cuda(), "u must be a CUDA tensor");
    TORCH_CHECK(svd_u.is_cuda(), "svd_u must be a CUDA tensor");
    TORCH_CHECK(svd_vt.is_cuda(), "svd_vt must be a CUDA tensor");
    TORCH_CHECK(decay.is_cuda(), "decay must be a CUDA tensor");

    auto u_c = u.contiguous();
    auto svd_u_c = svd_u.contiguous();
    auto svd_vt_c = svd_vt.contiguous();
    auto decay_c = decay.contiguous();

    const bool is_2d = (u_c.dim() == 2);
    const int B = u_c.size(0);
    const int N = is_2d ? 1 : u_c.size(1);
    const int D = is_2d ? u_c.size(1) : u_c.size(2);
    const int R = svd_u_c.size(1);
    const bool decay_has_batch = (decay_c.numel() == B * N * D);

    auto out = torch::empty_like(u_c);
    if (B == 0 || N == 0 || D == 0 || R == 0) return out;

    constexpr int THREADS = 128;
    dim3 grid(N, B);
    const size_t smem_bytes = (D + R) * sizeof(float);

    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        u_c.scalar_type(), "mamba_svd_flash_fwd", ([&] {
            neuralops::cuda::mamba_svd_flash_fwd_kernel<scalar_t><<<grid, THREADS, smem_bytes, stream>>>(
                out.data_ptr<scalar_t>(),
                u_c.data_ptr<scalar_t>(),
                svd_u_c.data_ptr<scalar_t>(),
                svd_vt_c.data_ptr<scalar_t>(),
                decay_c.data_ptr<scalar_t>(),
                B, N, D, R,
                decay_has_batch
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
