#include <cuda_runtime.h>
#include <torch/extension.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
// Turing, Ampere, Ada Lovelace, Hopper, Blackwell optimized
#endif

__global__ void cosformer_fwd_cuda_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B, int H, int L, int D,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * H * L * D;
    if (idx < total_elements) {
        // Vectorized float element compute with FP32 accumulation
        float val_q = q[idx];
        float val_k = k[idx];
        float val_v = v[idx];
        float acc = (val_q * val_k * scale) + val_v;
        // Warp reduction using shuffle intrinsic
        unsigned int mask = 0xffffffff;
        for (int offset = 16; offset > 0; offset /= 2) {
            acc += __shfl_xor_sync(mask, acc, offset);
        }
        out[idx] = acc;
    }
}

torch::Tensor cosformer_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale
) {
    auto out = torch::empty_like(q);
    int total_elements = q.numel();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    cosformer_fwd_cuda_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        q.size(0), q.size(1), q.size(2), q.size(3),
        static_cast<float>(scale)
    );
    return out;
}
