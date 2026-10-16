#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void paged_kv_compact_kernel(scalar_t* k_out, scalar_t* v_out, const scalar_t* k_in, const scalar_t* v_in, const int64_t* keep_indices, int num_keep, int D) {
    int pid = blockIdx.x;
    if (pid >= num_keep) return;
    int target_idx = keep_indices[pid];
    int tid = threadIdx.x;
    for (int d = tid; d < D; d += blockDim.x) {
        k_out[pid * D + d] = k_in[target_idx * D + d];
        v_out[pid * D + d] = v_in[target_idx * D + d];
    }
}
}}

std::vector<torch::Tensor> paged_kv_compact_fwd(const torch::Tensor& k_in, const torch::Tensor& v_in, const torch::Tensor& keep_indices) {
    CHECK_INPUT(k_in); CHECK_INPUT(v_in); CHECK_INPUT(keep_indices);
    int num_keep = keep_indices.size(0);
    int D = k_in.size(-1);
    auto k_out = torch::empty({num_keep, D}, k_in.options());
    auto v_out = torch::empty({num_keep, D}, v_in.options());
    int threads = std::min(512, ((D + 31) / 32) * 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto k_c = k_in.contiguous(); auto v_c = v_in.contiguous(); auto idx_c = keep_indices.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, k_in.scalar_type(), "paged_kv_compact_fwd", ([&] {
        neuralops::cuda::paged_kv_compact_kernel<scalar_t><<<num_keep, threads, 0, stream>>>(
            k_out.data_ptr<scalar_t>(), v_out.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(), idx_c.data_ptr<int64_t>(), num_keep, D);
    }));
    return {k_out, v_out};
}
