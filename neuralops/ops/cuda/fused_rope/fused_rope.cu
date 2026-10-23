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

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__launch_bounds__(128)
__global__ void fused_rope_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const int B,
    const int H,
    const int N,
    const int D,
    const int half_d,
    const int stride_xb,
    const int stride_xh,
    const int stride_xn,
    const int stride_xd,
    const int stride_cb,
    const int stride_ch,
    const int stride_cn,
    const int stride_cd,
    const int stride_sb,
    const int stride_sh,
    const int stride_sn,
    const int stride_sd,
    const int stride_ob,
    const int stride_oh,
    const int stride_on,
    const int stride_od
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int n = blockIdx.z;

    if (b >= B || h >= H || n >= N) return;

    const int tid = threadIdx.x;
    if (tid >= half_d) return;

    const int x1_idx = b * stride_xb + h * stride_xh + n * stride_xn + tid * stride_xd;
    const int x2_idx = b * stride_xb + h * stride_xh + n * stride_xn + (tid + half_d) * stride_xd;

    const int c_idx = b * stride_cb + h * stride_ch + n * stride_cn + tid * stride_cd;
    const int s_idx = b * stride_sb + h * stride_sh + n * stride_sn + tid * stride_sd;

    const float x1 = to_float(x[x1_idx]);
    const float x2 = to_float(x[x2_idx]);
    const float c = to_float(cos[c_idx]);
    const float s = to_float(sin[s_idx]);

    const float o1 = x1 * c - x2 * s;
    const float o2 = x2 * c + x1 * s;

    const int o1_idx = b * stride_ob + h * stride_oh + n * stride_on + tid * stride_od;
    const int o2_idx = b * stride_ob + h * stride_oh + n * stride_on + (tid + half_d) * stride_od;

    out[o1_idx] = from_float<scalar_t>(o1);
    out[o2_idx] = from_float<scalar_t>(o2);
}

template <typename scalar_t>
__launch_bounds__(128)
__global__ void fused_rope_bwd_kernel(
    scalar_t* __restrict__ dx,
    const scalar_t* __restrict__ dout,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    const int B,
    const int H,
    const int N,
    const int D,
    const int half_d,
    const int stride_xb,
    const int stride_xh,
    const int stride_xn,
    const int stride_xd,
    const int stride_cb,
    const int stride_ch,
    const int stride_cn,
    const int stride_cd,
    const int stride_sb,
    const int stride_sh,
    const int stride_sn,
    const int stride_sd,
    const int stride_ob,
    const int stride_oh,
    const int stride_on,
    const int stride_od
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int n = blockIdx.z;

    if (b >= B || h >= H || n >= N) return;

    const int tid = threadIdx.x;
    if (tid >= half_d) return;

    const int do1_idx = b * stride_ob + h * stride_oh + n * stride_on + tid * stride_od;
    const int do2_idx = b * stride_ob + h * stride_oh + n * stride_on + (tid + half_d) * stride_od;

    const int c_idx = b * stride_cb + h * stride_ch + n * stride_cn + tid * stride_cd;
    const int s_idx = b * stride_sb + h * stride_sh + n * stride_sn + tid * stride_sd;

    const float do1 = to_float(dout[do1_idx]);
    const float do2 = to_float(dout[do2_idx]);
    const float c = to_float(cos[c_idx]);
    const float s = to_float(sin[s_idx]);

    const float dx1 = do1 * c + do2 * s;
    const float dx2 = do2 * c - do1 * s;

    const int dx1_idx = b * stride_xb + h * stride_xh + n * stride_xn + tid * stride_xd;
    const int dx2_idx = b * stride_xb + h * stride_xh + n * stride_xn + (tid + half_d) * stride_xd;

    dx[dx1_idx] = from_float<scalar_t>(dx1);
    dx[dx2_idx] = from_float<scalar_t>(dx2);
}

}
}

torch::Tensor fused_rope_fwd(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin
) {
    CHECK_INPUT(x);
    CHECK_INPUT(cos);
    CHECK_INPUT(sin);

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);
    const int half_d = D / 2;

    auto out = torch::empty_like(x);
    const dim3 grid(B, H, N);
    const int threads = (half_d <= 32) ? 32 : ((half_d <= 64) ? 64 : 128);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto cos_b = cos.expand_as(x.narrow(3, 0, half_d));
    auto sin_b = sin.expand_as(x.narrow(3, 0, half_d));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fused_rope_fwd", ([&] {
            neuralops::cuda::fused_rope_fwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), cos_b.data_ptr<scalar_t>(), sin_b.data_ptr<scalar_t>(),
                B, H, N, D, half_d,
                x.stride(0), x.stride(1), x.stride(2), x.stride(3),
                cos_b.stride(0), cos_b.stride(1), cos_b.stride(2), cos_b.stride(3),
                sin_b.stride(0), sin_b.stride(1), sin_b.stride(2), sin_b.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor fused_rope_bwd(
    const torch::Tensor& dout,
    const torch::Tensor& cos,
    const torch::Tensor& sin
) {
    CHECK_INPUT(dout);
    CHECK_INPUT(cos);
    CHECK_INPUT(sin);

    const int B = dout.size(0);
    const int H = dout.size(1);
    const int N = dout.size(2);
    const int D = dout.size(3);
    const int half_d = D / 2;

    auto dx = torch::empty_like(dout);
    const dim3 grid(B, H, N);
    const int threads = (half_d <= 32) ? 32 : ((half_d <= 64) ? 64 : 128);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto cos_b = cos.expand_as(dout.narrow(3, 0, half_d));
    auto sin_b = sin.expand_as(dout.narrow(3, 0, half_d));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        dout.scalar_type(), "fused_rope_bwd", ([&] {
            neuralops::cuda::fused_rope_bwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                dx.data_ptr<scalar_t>(), dout.data_ptr<scalar_t>(), cos_b.data_ptr<scalar_t>(), sin_b.data_ptr<scalar_t>(),
                B, H, N, D, half_d,
                dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
                cos_b.stride(0), cos_b.stride(1), cos_b.stride(2), cos_b.stride(3),
                sin_b.stride(0), sin_b.stride(1), sin_b.stride(2), sin_b.stride(3),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return dx;
}
