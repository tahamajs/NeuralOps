// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t, int MAX_DK>
__launch_bounds__(128)
__global__ void recurrent_linear_step_vectorized_kernel(
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ kv_state,
    scalar_t* __restrict__ z_state,
    const scalar_t* __restrict__ q_t,
    const scalar_t* __restrict__ k_t,
    const scalar_t* __restrict__ v_t,
    const int B,
    const int H,
    const int Dk,
    const int Dv,
    const float eps
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int tid = threadIdx.x;

    const int qk_offset = bh * Dk;
    const int z_offset = bh * Dk;

    __shared__ float s_q[MAX_DK];
    __shared__ float s_k[MAX_DK];
    __shared__ float s_z[MAX_DK];

    for (int k = tid; k < Dk && k < MAX_DK; k += blockDim.x) {
        s_q[k] = to_float(q_t[qk_offset + k]);
        const float kval = to_float(k_t[qk_offset + k]);
        s_k[k] = kval;
        const float z_new = to_float(z_state[z_offset + k]) + kval;
        s_z[k] = z_new;
        z_state[z_offset + k] = from_float<scalar_t>(z_new);
    }
    __syncthreads();

    float denom_acc = 0.0f;
    #pragma unroll 4
    for (int k = tid; k < Dk && k < MAX_DK; k += blockDim.x) {
        denom_acc = fmaf(s_q[k], s_z[k], denom_acc);
    }
    denom_acc = block_reduce_sum(denom_acc);
    __shared__ float s_denom;
    if (tid == 0) {
        s_denom = denom_acc + eps;
    }
    __syncthreads();

    const float inv_denom = 1.0f / (s_denom != 0.0f ? s_denom : 1e-6f);

    const int v_offset = bh * Dv;
    const int kv_offset = bh * (Dk * Dv);
    const int out_offset = bh * Dv;

    for (int v = tid; v < Dv; v += blockDim.x) {
        const float v_val = to_float(v_t[v_offset + v]);
        float num_acc = 0.0f;

        #pragma unroll 4
        for (int k = 0; k < Dk && k < MAX_DK; ++k) {
            const int idx = kv_offset + k * Dv + v;
            const float kv_val = fmaf(s_k[k], v_val, to_float(kv_state[idx]));
            kv_state[idx] = from_float<scalar_t>(kv_val);
            num_acc = fmaf(s_q[k], kv_val, num_acc);
        }

        out[out_offset + v] = from_float<scalar_t>(num_acc * inv_denom);
    }
}

}
}

std::vector<torch::Tensor> recurrent_linear_step_fwd(
    const torch::Tensor& q_t,
    const torch::Tensor& k_t,
    const torch::Tensor& v_t,
    torch::Tensor& kv_state,
    torch::Tensor& z_state,
    const float eps
) {
    TORCH_CHECK(q_t.is_cuda(), "q_t must be a CUDA tensor");
    TORCH_CHECK(k_t.is_cuda(), "k_t must be a CUDA tensor");
    TORCH_CHECK(v_t.is_cuda(), "v_t must be a CUDA tensor");

    const int B = q_t.size(0);
    const int H = q_t.size(1);
    const int Dk = q_t.size(2);
    const int Dv = v_t.size(2);
    TORCH_CHECK(Dk <= 128, "state_update_fwd: Dk=", Dk, " > 128 not supported");

    auto out = torch::empty({B, H, Dv}, q_t.options());

    const dim3 grid(1, B * H);
    const int threads = 128;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q_t.contiguous();
    auto k_c = k_t.contiguous();
    auto v_c = v_t.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_t.scalar_type(), "recurrent_linear_step_fwd", ([&] {
            if (Dk <= 64) {
                neuralops::cuda::recurrent_linear_step_vectorized_kernel<scalar_t, 64><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    kv_state.data_ptr<scalar_t>(),
                    z_state.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, Dk, Dv, eps
                );
            } else {
                neuralops::cuda::recurrent_linear_step_vectorized_kernel<scalar_t, 128><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    kv_state.data_ptr<scalar_t>(),
                    z_state.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, Dk, Dv, eps
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {out, kv_state, z_state};
}
