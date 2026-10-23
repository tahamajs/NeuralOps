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
#include <math.h>
#include <c10/util/Optional.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t, int MAX_DK>
__launch_bounds__(128)
__global__ void recurrent_gla_step_kernel(
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ s_next,
    const scalar_t* __restrict__ q_t,
    const scalar_t* __restrict__ k_t,
    const scalar_t* __restrict__ v_t,
    const scalar_t* __restrict__ s_prev,
    const scalar_t* __restrict__ alpha_t,
    const int B,
    const int H,
    const int Dk,
    const int Dv,
    const float scale
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int tid = threadIdx.x;
    const float a_val = to_float(alpha_t[bh]);

    __shared__ float s_q[MAX_DK];
    __shared__ float s_k[MAX_DK];

    const int qk_offset = bh * Dk;
    for (int k = tid; k < Dk && k < MAX_DK; k += blockDim.x) {
        s_q[k] = to_float(q_t[qk_offset + k]);
        s_k[k] = to_float(k_t[qk_offset + k]);
    }
    __syncthreads();

    const int v_offset = bh * Dv;
    const int s_offset = bh * (Dk * Dv);
    const int out_offset = bh * Dv;

    for (int v = tid; v < Dv; v += blockDim.x) {
        const float v_val = to_float(v_t[v_offset + v]);
        float y_acc = 0.0f;

        #pragma unroll 4
        for (int k = 0; k < Dk && k < MAX_DK; ++k) {
            const int idx = s_offset + k * Dv + v;
            const float s_old = to_float(s_prev[idx]);
            const float s_val = a_val * s_old + s_k[k] * v_val;
            s_next[idx] = from_float<scalar_t>(s_val);
            y_acc += s_q[k] * s_val;
        }

        out[out_offset + v] = from_float<scalar_t>(y_acc * scale);
    }
}

template <typename scalar_t, int MAX_DK>
__launch_bounds__(128)
__global__ void recurrent_gla_step_bwd_kernel(
    scalar_t* __restrict__ dq_t,
    scalar_t* __restrict__ dk_t,
    scalar_t* __restrict__ dv_t,
    scalar_t* __restrict__ ds_prev,
    scalar_t* __restrict__ dalpha_t,
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ ds_next,
    const scalar_t* __restrict__ q_t,
    const scalar_t* __restrict__ k_t,
    const scalar_t* __restrict__ v_t,
    const scalar_t* __restrict__ s_prev,
    const scalar_t* __restrict__ alpha_t,
    const int B,
    const int H,
    const int Dk,
    const int Dv,
    const float scale,
    const bool has_ds_next
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int tid = threadIdx.x;
    const float a_val = to_float(alpha_t[bh]);

    __shared__ float s_q[MAX_DK];
    __shared__ float s_k[MAX_DK];
    __shared__ float s_dq_acc[MAX_DK];
    __shared__ float s_dk_acc[MAX_DK];
    __shared__ float s_dalpha_warp[32];

    const int qk_offset = bh * Dk;
    for (int k = tid; k < Dk && k < MAX_DK; k += blockDim.x) {
        s_q[k] = to_float(q_t[qk_offset + k]);
        s_k[k] = to_float(k_t[qk_offset + k]);
        s_dq_acc[k] = 0.0f;
        s_dk_acc[k] = 0.0f;
    }
    if (tid < 32) s_dalpha_warp[tid] = 0.0f;
    __syncthreads();

    const int v_offset = bh * Dv;
    const int s_offset = bh * (Dk * Dv);

    float thread_dalpha = 0.0f;

    for (int v = tid; v < Dv; v += blockDim.x) {
        const float dout_v = to_float(dout[v_offset + v]);
        const float v_val = to_float(v_t[v_offset + v]);
        float dv_acc = 0.0f;

        #pragma unroll 4
        for (int k = 0; k < Dk && k < MAX_DK; ++k) {
            const int idx = s_offset + k * Dv + v;
            const float s_old = to_float(s_prev[idx]);
            const float s_new = a_val * s_old + s_k[k] * v_val;

            atomicAdd(&s_dq_acc[k], scale * dout_v * s_new);

            float ds_new_val = scale * dout_v * s_q[k];
            if (has_ds_next) {
                ds_new_val += to_float(ds_next[idx]);
            }

            ds_prev[idx] = from_float<scalar_t>(ds_new_val * a_val);
            atomicAdd(&s_dk_acc[k], ds_new_val * v_val);
            dv_acc += ds_new_val * s_k[k];
            thread_dalpha += ds_new_val * s_old;
        }

        dv_t[v_offset + v] = from_float<scalar_t>(dv_acc);
    }
    __syncthreads();

    for (int k = tid; k < Dk && k < MAX_DK; k += blockDim.x) {
        dq_t[qk_offset + k] = from_float<scalar_t>(s_dq_acc[k]);
        dk_t[qk_offset + k] = from_float<scalar_t>(s_dk_acc[k]);
    }

    thread_dalpha = warp_reduce_sum(thread_dalpha);
    const int lane = tid % 32;
    const int wid = tid / 32;
    if (lane == 0) {
        s_dalpha_warp[wid] = thread_dalpha;
    }
    __syncthreads();

    if (wid == 0) {
        float b_dalpha = (tid < (blockDim.x + 31) / 32) ? s_dalpha_warp[tid] : 0.0f;
        b_dalpha = warp_reduce_sum(b_dalpha);
        if (tid == 0) {
            dalpha_t[bh] = from_float<scalar_t>(b_dalpha);
        }
    }
}

}
}

std::vector<torch::Tensor> recurrent_gla_step_fwd(
    const torch::Tensor& q_t,
    const torch::Tensor& k_t,
    const torch::Tensor& v_t,
    torch::Tensor& s_state,
    const torch::Tensor& alpha_t
) {
    TORCH_CHECK(q_t.is_cuda(), "q_t must be a CUDA tensor");
    TORCH_CHECK(k_t.is_cuda(), "k_t must be a CUDA tensor");
    TORCH_CHECK(v_t.is_cuda(), "v_t must be a CUDA tensor");

    const int B = q_t.size(0);
    const int H = q_t.size(1);
    const int Dk = q_t.size(2);
    const int Dv = v_t.size(2);

    auto out = torch::empty({B, H, Dv}, q_t.options());
    const float scale = 1.0f / sqrtf(static_cast<float>(Dk));

    const dim3 grid(1, B * H);
    const int threads = 128;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q_t.contiguous();
    auto k_c = k_t.contiguous();
    auto v_c = v_t.contiguous();
    auto s_c = s_state.contiguous();
    auto a_c = alpha_t.reshape({-1}).contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_t.scalar_type(), "recurrent_gla_step_fwd", ([&] {
            if (Dk <= 64) {
                neuralops::cuda::recurrent_gla_step_kernel<scalar_t, 64><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    s_state.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    s_c.data_ptr<scalar_t>(),
                    a_c.data_ptr<scalar_t>(),
                    B, H, Dk, Dv, scale
                );
            } else {
                neuralops::cuda::recurrent_gla_step_kernel<scalar_t, 128><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    s_state.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    s_c.data_ptr<scalar_t>(),
                    a_c.data_ptr<scalar_t>(),
                    B, H, Dk, Dv, scale
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {out, s_state};
}

std::vector<torch::Tensor> recurrent_gla_step_bwd(
    const torch::Tensor& dout,
    const c10::optional<torch::Tensor>& ds_next,
    const torch::Tensor& q_t,
    const torch::Tensor& k_t,
    const torch::Tensor& v_t,
    const torch::Tensor& s_prev,
    const torch::Tensor& alpha_t
) {
    TORCH_CHECK(dout.is_cuda(), "dout must be a CUDA tensor");
    TORCH_CHECK(q_t.is_cuda(), "q_t must be a CUDA tensor");
    TORCH_CHECK(k_t.is_cuda(), "k_t must be a CUDA tensor");
    TORCH_CHECK(v_t.is_cuda(), "v_t must be a CUDA tensor");
    TORCH_CHECK(s_prev.is_cuda(), "s_prev must be a CUDA tensor");
    TORCH_CHECK(alpha_t.is_cuda(), "alpha_t must be a CUDA tensor");

    const int B = q_t.size(0);
    const int H = q_t.size(1);
    const int Dk = q_t.size(2);
    const int Dv = v_t.size(2);

    auto dq = torch::empty_like(q_t);
    auto dk = torch::empty_like(k_t);
    auto dv = torch::empty_like(v_t);
    auto ds_prev_out = torch::empty_like(s_prev);
    auto dalpha = torch::empty_like(alpha_t);

    const float scale = 1.0f / sqrtf(static_cast<float>(Dk));

    const dim3 grid(1, B * H);
    const int threads = 128;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto dout_c = dout.contiguous();
    auto q_c = q_t.contiguous();
    auto k_c = k_t.contiguous();
    auto v_c = v_t.contiguous();
    auto s_c = s_prev.contiguous();
    auto a_c = alpha_t.reshape({-1}).contiguous();

    const bool has_ds = ds_next.has_value() && ds_next.value().defined();
    torch::Tensor ds_c;
    if (has_ds) {
        ds_c = ds_next.value().contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q_t.scalar_type(), "recurrent_gla_step_bwd", ([&] {
            const scalar_t* ds_ptr = has_ds ? ds_c.data_ptr<scalar_t>() : nullptr;
            if (Dk <= 64) {
                neuralops::cuda::recurrent_gla_step_bwd_kernel<scalar_t, 64><<<grid, threads, 0, stream>>>(
                    dq.data_ptr<scalar_t>(),
                    dk.data_ptr<scalar_t>(),
                    dv.data_ptr<scalar_t>(),
                    ds_prev_out.data_ptr<scalar_t>(),
                    dalpha.data_ptr<scalar_t>(),
                    dout_c.data_ptr<scalar_t>(),
                    ds_ptr,
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    s_c.data_ptr<scalar_t>(),
                    a_c.data_ptr<scalar_t>(),
                    B, H, Dk, Dv, scale,
                    has_ds
                );
            } else {
                neuralops::cuda::recurrent_gla_step_bwd_kernel<scalar_t, 128><<<grid, threads, 0, stream>>>(
                    dq.data_ptr<scalar_t>(),
                    dk.data_ptr<scalar_t>(),
                    dv.data_ptr<scalar_t>(),
                    ds_prev_out.data_ptr<scalar_t>(),
                    dalpha.data_ptr<scalar_t>(),
                    dout_c.data_ptr<scalar_t>(),
                    ds_ptr,
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    s_c.data_ptr<scalar_t>(),
                    a_c.data_ptr<scalar_t>(),
                    B, H, Dk, Dv, scale,
                    has_ds
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {dq, dk, dv, ds_prev_out, dalpha};
}
