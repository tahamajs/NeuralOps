#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {
template <typename scalar_t>
__global__ void paged_recurrent_step_kernel(scalar_t* out, const scalar_t* q, const scalar_t* k, const scalar_t* v, scalar_t* page_pool, const int64_t* block_tables, int B, int H, int D, float decay) {
    int pid_b = blockIdx.x;
    int pid_h = blockIdx.y;
    if (pid_b >= B || pid_h >= H) return;
    int phys_page = block_tables[pid_b];
    int tid = threadIdx.x;
    for (int d2 = tid; d2 < D; d2 += blockDim.x) {
        float out_acc = 0.0f;
        for (int d1 = 0; d1 < D; ++d1) {
            int s_idx = phys_page * (H * D * D) + pid_h * (D * D) + d1 * D + d2;
            float s_prev = to_float(page_pool[s_idx]);
            float kv = to_float(k[pid_b * (H * D) + pid_h * D + d1]) * to_float(v[pid_b * (H * D) + pid_h * D + d2]);
            float s_next = decay * s_prev + kv;
            page_pool[s_idx] = from_float<scalar_t>(s_next);
            out_acc += to_float(q[pid_b * (H * D) + pid_h * D + d1]) * s_next;
        }
        out[pid_b * (H * D) + pid_h * D + d2] = from_float<scalar_t>(out_acc);
    }
}
}}

torch::Tensor paged_recurrent_step_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, torch::Tensor& page_pool, const torch::Tensor& block_tables, float decay) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v); CHECK_INPUT(page_pool); CHECK_INPUT(block_tables);
    int B = q.size(0); int H = q.size(1); int D = q.size(2);
    auto out = torch::empty_like(q);
    dim3 blocks(B, H);
    int threads = std::min(256, ((D + 31) / 32) * 32);
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous(); auto k_c = k.contiguous(); auto v_c = v.contiguous(); auto bt_c = block_tables.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "paged_recurrent_step_fwd", ([&] {
        neuralops::cuda::paged_recurrent_step_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            out.data_ptr<scalar_t>(), q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(), page_pool.data_ptr<scalar_t>(), bt_c.data_ptr<int64_t>(), B, H, D, decay);
    }));
    return out;
}
