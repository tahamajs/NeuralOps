#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void bitonic_topk_kernel(int64_t* out_indices, const scalar_t* x, int rows, int N, int K) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int tid = threadIdx.x;
    extern __shared__ int64_t s_indices[];
    for (int i = tid; i < N; i += blockDim.x) s_indices[i] = i;
    __syncthreads();
    // Bitonic sort on indices by x values
    for (int k = 2; k <= N; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = tid; i < N; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i && ixj < N) {
                    if ((i & k) == 0) {
                        if (to_float(x[r * N + s_indices[i]]) < to_float(x[r * N + s_indices[ixj]])) {
                            int64_t temp = s_indices[i]; s_indices[i] = s_indices[ixj]; s_indices[ixj] = temp;
                        }
                    } else {
                        if (to_float(x[r * N + s_indices[i]]) > to_float(x[r * N + s_indices[ixj]])) {
                            int64_t temp = s_indices[i]; s_indices[i] = s_indices[ixj]; s_indices[ixj] = temp;
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
}}

torch::Tensor bitonic_topk_indices_fwd(const torch::Tensor& x, int K) {
    CHECK_INPUT(x);
    int N = x.size(-1);
    int rows = x.numel() / N;
    auto out = torch::empty({rows, K}, x.options().dtype(torch::kInt64));
    int threads = std::min(512, ((N + 31) / 32) * 32);
    size_t shared_mem = N * sizeof(int64_t);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "bitonic_topk_indices_fwd", ([&] {
        neuralops::cuda::bitonic_topk_kernel<scalar_t><<<rows, threads, shared_mem, stream>>>(
            out.data_ptr<int64_t>(), x_c.data_ptr<scalar_t>(), rows, N, K);
    }));
    return out;
}
