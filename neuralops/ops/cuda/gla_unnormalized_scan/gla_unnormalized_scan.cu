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

template <typename scalar_t, bool G_HAS_D>
__global__ void __launch_bounds__(256) gla_unnorm_scan_kernel_vec4(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ g,
    const int B,
    const int H,
    const int N,
    const int D
) {
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;
    const int d4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (b >= B || h >= H || d4 >= D) return;

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    const int base_bh = (b * H + h) * N * D + d4;
    const int g_base_bh = (b * H + h) * N;

    for (int t = 0; t < N; ++t) {
        const int idx = base_bh + t * D;

        float gate0, gate1, gate2, gate3;
        if constexpr (G_HAS_D) {
            gate0 = expf(to_float(g[idx + 0]));
            gate1 = expf(to_float(g[idx + 1]));
            gate2 = expf(to_float(g[idx + 2]));
            gate3 = expf(to_float(g[idx + 3]));
        } else {
            const float common_gate = expf(to_float(g[g_base_bh + t]));
            gate0 = common_gate;
            gate1 = common_gate;
            gate2 = common_gate;
            gate3 = common_gate;
        }

        float k0 = to_float(k[idx + 0]);
        float k1 = to_float(k[idx + 1]);
        float k2 = to_float(k[idx + 2]);
        float k3 = to_float(k[idx + 3]);

        float v0 = to_float(v[idx + 0]);
        float v1 = to_float(v[idx + 1]);
        float v2 = to_float(v[idx + 2]);
        float v3 = to_float(v[idx + 3]);

        s0 = gate0 * s0 + k0 * v0;
        s1 = gate1 * s1 + k1 * v1;
        s2 = gate2 * s2 + k2 * v2;
        s3 = gate3 * s3 + k3 * v3;

        float q0 = to_float(q[idx + 0]);
        float q1 = to_float(q[idx + 1]);
        float q2 = to_float(q[idx + 2]);
        float q3 = to_float(q[idx + 3]);

        out[idx + 0] = from_float<scalar_t>(q0 * s0);
        out[idx + 1] = from_float<scalar_t>(q1 * s1);
        out[idx + 2] = from_float<scalar_t>(q2 * s2);
        out[idx + 3] = from_float<scalar_t>(q3 * s3);
    }
}

template <typename scalar_t, bool G_HAS_D>
__global__ void __launch_bounds__(256) gla_unnorm_scan_kernel_scalar(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ g,
    const int B,
    const int H,
    const int N,
    const int D
) {
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || h >= H || d >= D) return;

    float s = 0.0f;
    const int base_bh = (b * H + h) * N * D + d;
    const int g_base_bh = (b * H + h) * N;

    for (int t = 0; t < N; ++t) {
        const int idx = base_bh + t * D;
        const float gate = G_HAS_D ? expf(to_float(g[idx])) : expf(to_float(g[g_base_bh + t]));
        const float k_val = to_float(k[idx]);
        const float v_val = to_float(v[idx]);

        s = gate * s + k_val * v_val;
        const float q_val = to_float(q[idx]);
        out[idx] = from_float<scalar_t>(q_val * s);
    }
}

}}

torch::Tensor gla_unnormalized_scan_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(g);

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    auto out = torch::empty_like(q);

    if (B == 0 || H == 0 || D == 0 || N == 0) return out;

    const bool g_has_d = (g.dim() == 4 && g.size(3) == D);

    constexpr int THREADS = 256;
    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto g_c = g.contiguous();

    if (D % 4 == 0) {
        const int num_vec4 = D / 4;
        const int blocks_x = (num_vec4 + THREADS - 1) / THREADS;
        dim3 grid(blocks_x, B * H);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            q.scalar_type(), "gla_unnormalized_scan_fwd_vec4", ([&] {
                if (g_has_d) {
                    neuralops::cuda::gla_unnorm_scan_kernel_vec4<scalar_t, true><<<grid, THREADS, 0, stream>>>(
                        out.data_ptr<scalar_t>(),
                        q_c.data_ptr<scalar_t>(),
                        k_c.data_ptr<scalar_t>(),
                        v_c.data_ptr<scalar_t>(),
                        g_c.data_ptr<scalar_t>(),
                        B, H, N, D
                    );
                } else {
                    neuralops::cuda::gla_unnorm_scan_kernel_vec4<scalar_t, false><<<grid, THREADS, 0, stream>>>(
                        out.data_ptr<scalar_t>(),
                        q_c.data_ptr<scalar_t>(),
                        k_c.data_ptr<scalar_t>(),
                        v_c.data_ptr<scalar_t>(),
                        g_c.data_ptr<scalar_t>(),
                        B, H, N, D
                    );
                }
            })
        );
    } else {
        const int blocks_x = (D + THREADS - 1) / THREADS;
        dim3 grid(blocks_x, B * H);

        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            q.scalar_type(), "gla_unnormalized_scan_fwd_scalar", ([&] {
                if (g_has_d) {
                    neuralops::cuda::gla_unnorm_scan_kernel_scalar<scalar_t, true><<<grid, THREADS, 0, stream>>>(
                        out.data_ptr<scalar_t>(),
                        q_c.data_ptr<scalar_t>(),
                        k_c.data_ptr<scalar_t>(),
                        v_c.data_ptr<scalar_t>(),
                        g_c.data_ptr<scalar_t>(),
                        B, H, N, D
                    );
                } else {
                    neuralops::cuda::gla_unnorm_scan_kernel_scalar<scalar_t, false><<<grid, THREADS, 0, stream>>>(
                        out.data_ptr<scalar_t>(),
                        q_c.data_ptr<scalar_t>(),
                        k_c.data_ptr<scalar_t>(),
                        v_c.data_ptr<scalar_t>(),
                        g_c.data_ptr<scalar_t>(),
                        B, H, N, D
                    );
                }
            })
        );
    }

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
