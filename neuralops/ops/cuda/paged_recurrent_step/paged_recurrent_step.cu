// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(256) paged_recurrent_step_kernel(
    scalar_t* __restrict__ page_pool,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ block_tables,
    const int B,
    const int H,
    const int D,
    const int page_slot,
    const int max_pages,
    const float decay
) {
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;
    const int d2 = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || h >= H || d2 >= D) return;

    const int64_t phys_page = (max_pages > 1) ? block_tables[b * max_pages + page_slot] : block_tables[b];
    const int page_stride = H * D * D;
    const int head_stride = D * D;

    float out_val = 0.0f;
    const float v_val = to_float(v[(b * H + h) * D + d2]);

    for (int d1 = 0; d1 < D; ++d1) {
        const int s_idx = phys_page * page_stride + h * head_stride + d1 * D + d2;
        const float s_old = to_float(page_pool[s_idx]);
        const float k_val = to_float(k[(b * H + h) * D + d1]);
        const float q_val = to_float(q[(b * H + h) * D + d1]);

        const float s_new = decay * s_old + k_val * v_val;
        page_pool[s_idx] = from_float<scalar_t>(s_new);
        out_val += q_val * s_new;
    }

    out[(b * H + h) * D + d2] = from_float<scalar_t>(out_val);
}

}}

torch::Tensor paged_recurrent_step_fwd(
    torch::Tensor& page_pool,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& block_tables,
    int page_slot = 0,
    float decay = 1.0f
) {

    CHECK_INPUT(page_pool);
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(block_tables);

    const int B = q.size(0);
    const int H = q.size(1);
    const int D = q.size(2);
    const int max_pages = (block_tables.dim() > 1) ? block_tables.size(1) : 1;

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto bt_c = block_tables.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "paged_recurrent_step_fwd", ([&] {
            neuralops::cuda::paged_recurrent_step_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                page_pool.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                bt_c.data_ptr<int64_t>(),
                B, H, D, page_slot, max_pages, decay
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
