#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Fast Walsh-Hadamard Transform in shared memory
template <typename scalar_t>
__global__ void __launch_bounds__(256) fwht_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const int total_rows,
    const int B_dim
) {
    const int row = blockIdx.y;
    if (row >= total_rows) return;

    extern __shared__ float s_mem[];
    const int tid = threadIdx.x;

    // Load into shared memory with zero-padding
    for (int i = tid; i < B_dim; i += blockDim.x) {
        s_mem[i] = to_float(x[row * B_dim + i]);
    }
    __syncthreads();

    // In-shared-memory butterfly rounds
    for (int len = 1; len < B_dim; len <<= 1) {
        for (int i = tid * (len << 1); i < B_dim; i += blockDim.x * (len << 1)) {
            for (int j = 0; j < len; ++j) {
                const float u = s_mem[i + j];
                const float v = s_mem[i + len + j];
                s_mem[i + j] = u + v;
                s_mem[i + len + j] = u - v;
            }
        }
        __syncthreads();
    }

    const float norm = rsqrtf(static_cast<float>(B_dim));
    for (int i = tid; i < B_dim; i += blockDim.x) {
        out[row * B_dim + i] = from_float<scalar_t>(s_mem[i] * norm);
    }
}

}} // namespace neuralops::cuda

torch::Tensor fwht_fwd(const torch::Tensor& x, int B_dim) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    const int total_elements = x.numel();
    const int total_rows = total_elements / B_dim;
    if (total_rows == 0 || B_dim == 0) return out;

    const int threads = std::min(256, B_dim);
    const size_t shmem_size = B_dim * sizeof(float);
    dim3 grid(1, total_rows);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fwht_fwd", ([&] {
            neuralops::cuda::fwht_kernel<scalar_t><<<grid, threads, shmem_size, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                total_rows,
                B_dim
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
