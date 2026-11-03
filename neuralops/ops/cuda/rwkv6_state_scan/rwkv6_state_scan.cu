// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <math.h>
#include "../_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__global__ void rwkv6_state_scan_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ u,
    const int B, const int H, const int T, const int D
) {
    const int bh = blockIdx.x; if (bh >= B * H) return;
    const int head_offset = bh * T * D; const int h = bh % H;

    // Vectorized 128-bit float4 path when scalar_t is float and D % 4 == 0
    if constexpr (std::is_same_v<scalar_t, float>) {
        if (D % 4 == 0) {
            const int num_vec = D / 4;
            const int vi = threadIdx.x;
            if (vi >= num_vec) return;
            const int d = vi * 4;

            const float4 u_raw = *reinterpret_cast<const float4*>(u + h * D + d);
            float4 u_val;
            u_val.x = expf(u_raw.x);
            u_val.y = expf(u_raw.y);
            u_val.z = expf(u_raw.z);
            u_val.w = expf(u_raw.w);

            float4 state = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            #pragma unroll 4
            for (int t = 0; t < T; ++t) {
                const int idx = head_offset + t * D + d;
                const float4 r_val = *reinterpret_cast<const float4*>(r + idx);
                const float4 k_val = *reinterpret_cast<const float4*>(k + idx);
                const float4 v_val = *reinterpret_cast<const float4*>(v + idx);
                const float4 w_val = *reinterpret_cast<const float4*>(w + idx);

                float4 decay;
                decay.x = expf(-expf(w_val.x));
                decay.y = expf(-expf(w_val.y));
                decay.z = expf(-expf(w_val.z));
                decay.w = expf(-expf(w_val.w));

                const float4 kv = make_float4(k_val.x * v_val.x, k_val.y * v_val.y, k_val.z * v_val.z, k_val.w * v_val.w);

                float4 y_val;
                y_val.x = r_val.x * (u_val.x * kv.x + state.x);
                y_val.y = r_val.y * (u_val.y * kv.y + state.y);
                y_val.z = r_val.z * (u_val.z * kv.z + state.z);
                y_val.w = r_val.w * (u_val.w * kv.w + state.w);

                *reinterpret_cast<float4*>(out + idx) = y_val;

                state.x = state.x * decay.x + kv.x;
                state.y = state.y * decay.y + kv.y;
                state.z = state.z * decay.z + kv.z;
                state.w = state.w * decay.w + kv.w;
            }
            return;
        }
    }

    const int d = threadIdx.x; if (d >= D) return;
    const float u_val = expf(static_cast<float>(u[h * D + d]));
    float state = 0.0f;
    for (int t = 0; t < T; ++t) {
        const int idx = head_offset + t * D + d;
        float r_val = static_cast<float>(r[idx]); float k_val = static_cast<float>(k[idx]);
        float v_val = static_cast<float>(v[idx]); float w_val = static_cast<float>(w[idx]);
        float decay = expf(-expf(w_val));
        float y_val = r_val * (u_val * k_val * v_val + state);
        out[idx] = static_cast<scalar_t>(y_val);
        state = state * decay + k_val * v_val;
    }
}

} // namespace cuda
} // namespace neuralops

torch::Tensor rwkv6_state_scan_fwd(
    const torch::Tensor& r, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& w, const torch::Tensor& u
) {
    CHECK_INPUT(r); CHECK_INPUT(k); CHECK_INPUT(v); CHECK_INPUT(w); CHECK_INPUT(u);
    const int B = r.size(0); const int H = r.size(1); const int T = r.size(2); const int D = r.size(3);
    auto out = torch::empty_like(v);
    dim3 grid(B * H); dim3 block(D > 1024 ? 1024 : D);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r.scalar_type(), "rwkv6_state_scan_fwd_cuda", ([&] {
        neuralops::cuda::rwkv6_state_scan_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(), r.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
            w.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), B, H, T, D
        );
    }));
    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
