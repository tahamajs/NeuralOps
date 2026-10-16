#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void fwht_kernel(scalar_t* out, const scalar_t* x, int total_rows, int B_dim) {
    int pid = blockIdx.x;
    if (pid >= total_rows) return;
    int tid = threadIdx.x;
    extern __shared__ float s_data[];
    if (tid < B_dim) s_data[tid] = to_float(x[pid * B_dim + tid]);
    __syncthreads();
    for (int h = 1; h < B_dim; h *= 2) {
        if ((tid & h) == 0 && (tid + h) < B_dim) {
            float u = s_data[tid];
            float v = s_data[tid + h];
            s_data[tid] = u + v;
            s_data[tid + h] = u - v;
        }
        __syncthreads();
    }
    float norm = rsqrtf((float)B_dim);
    if (tid < B_dim) out[pid * B_dim + tid] = from_float<scalar_t>(s_data[tid] * norm);
}
}}

torch::Tensor fwht_fwd(const torch::Tensor& x, int B_dim) {
    CHECK_INPUT(x);
    auto orig_shape = x.sizes();
    int total_rows = x.numel() / B_dim;
    auto out = torch::empty_like(x);
    int threads = B_dim;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    size_t shared_mem = B_dim * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "fwht_fwd", ([&] {
        neuralops::cuda::fwht_kernel<scalar_t><<<total_rows, threads, shared_mem, stream>>>(out.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(), total_rows, B_dim);
    }));
    return out;
}
