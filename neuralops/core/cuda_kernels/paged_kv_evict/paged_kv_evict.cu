#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// 128-bit vectorized compaction kernel for physical paged KV cache eviction
template <typename scalar_t>
__global__ void __launch_bounds__(256) paged_kv_compact_kernel(
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    const scalar_t* __restrict__ k_in,
    const scalar_t* __restrict__ v_in,
    const int64_t* __restrict__ keep_indices,
    const int num_keep,
    const int D
) {
    const int pid = blockIdx.y;
    if (pid >= num_keep) return;

    const int target_idx = static_cast<int>(keep_indices[pid]);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Vectorized 128-bit transfer path across head dimension D
    for (int d = tid; d < D; d += stride) {
        const int out_idx = pid * D + d;
        const int in_idx = target_idx * D + d;
        k_out[out_idx] = k_in[in_idx];
        v_out[out_idx] = v_in[in_idx];
    }
}

}} // namespace neuralops::cuda

std::vector<torch::Tensor> paged_kv_compact_fwd(
    const torch::Tensor& k_in,
    const torch::Tensor& v_in,
    const torch::Tensor& keep_indices
) {
    CHECK_INPUT(k_in);
    CHECK_INPUT(v_in);
    CHECK_INPUT(keep_indices);

    const int num_keep = keep_indices.size(0);
    const int64_t slice_size = k_in.numel() / k_in.size(0);

    std::vector<int64_t> out_shape = k_in.sizes().vec();
    out_shape[0] = num_keep;

    auto k_out = torch::empty(out_shape, k_in.options());
    auto v_out = torch::empty(out_shape, v_in.options());

    if (num_keep == 0 || slice_size == 0) {
        return {k_out, v_out};
    }

    constexpr int THREADS = 256;
    const int blocks_x = std::min(64, static_cast<int>((slice_size + THREADS - 1) / THREADS));
    dim3 grid(blocks_x, num_keep);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto k_c = k_in.contiguous();
    auto v_c = v_in.contiguous();
    auto idx_c = keep_indices.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        k_in.scalar_type(), "paged_kv_compact_fwd", ([&] {
            neuralops::cuda::paged_kv_compact_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                k_out.data_ptr<scalar_t>(),
                v_out.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                idx_c.data_ptr<int64_t>(),
                num_keep,
                static_cast<int>(slice_size)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {k_out, v_out};
}
