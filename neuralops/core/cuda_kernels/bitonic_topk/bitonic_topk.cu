#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// Bitonic Top-K Indices Kernel:
// Performs cooperative bitonic sorting over N in shared memory.
template <typename scalar_t>
__global__ void __launch_bounds__(512) bitonic_topk_kernel(
    int64_t* __restrict__ out_indices,
    const scalar_t* __restrict__ x,
    const int rows,
    const int N,
    const int K
) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x;

    extern __shared__ int64_t s_indices[];
    for (int i = tid; i < N; i += blockDim.x) {
        s_indices[i] = i;
    }
    __syncthreads();

    // Bitonic sort on indices by x values in descending order
    for (int k = 2; k <= N; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = tid; i < N; i += blockDim.x) {
                const int ixj = i ^ j;
                if (ixj > i && ixj < N) {
                    if ((i & k) == 0) {
                        if (to_float(x[r * N + s_indices[i]]) < to_float(x[r * N + s_indices[ixj]])) {
                            int64_t temp = s_indices[i];
                            s_indices[i] = s_indices[ixj];
                            s_indices[ixj] = temp;
                        }
                    } else {
                        if (to_float(x[r * N + s_indices[i]]) > to_float(x[r * N + s_indices[ixj]])) {
                            int64_t temp = s_indices[i];
                            s_indices[i] = s_indices[ixj];
                            s_indices[ixj] = temp;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int k = tid; k < K; k += blockDim.x) {
        out_indices[r * K + k] = s_indices[k];
    }
}

}} // namespace neuralops::cuda

torch::Tensor bitonic_topk_indices_fwd(const torch::Tensor& x, int K) {
    CHECK_INPUT(x);
    const int N = x.size(-1);
    const int rows = x.numel() / N;
    auto out = torch::empty({rows, K}, x.options().dtype(torch::kInt64));
    if (rows == 0 || N == 0 || K == 0) return out;

    int threads = std::min(512, ((N + 31) / 32) * 32);
    if (threads < 32) threads = 32;

    const size_t shared_mem = N * sizeof(int64_t);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "bitonic_topk_indices_fwd", ([&] {
            neuralops::cuda::bitonic_topk_kernel<scalar_t><<<rows, threads, shared_mem, stream>>>(
                out.data_ptr<int64_t>(),
                x_c.data_ptr<scalar_t>(),
                rows,
                N,
                K
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
