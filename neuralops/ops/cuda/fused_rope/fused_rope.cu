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

    // Vectorized 128-bit load fast path if memory is contiguous along D and half_d is divisible by 4
    if (stride_xd == 1 && stride_cd == 1 && stride_sd == 1 && stride_od == 1 && (half_d % 4 == 0) && std::is_same_v<scalar_t, float>) {
        const int num_vec = half_d / 4;
        const int x_base = b * stride_xb + h * stride_xh + n * stride_xn;
        const int c_base = b * stride_cb + h * stride_ch + n * stride_cn;
        const int s_base = b * stride_sb + h * stride_sh + n * stride_sn;
        const int o_base = b * stride_ob + h * stride_oh + n * stride_on;

        for (int vi = tid; vi < num_vec; vi += blockDim.x) {
            const int idx = vi * 4;
            const float4 x1_4 = *reinterpret_cast<const float4*>(x + x_base + idx);
            const float4 x2_4 = *reinterpret_cast<const float4*>(x + x_base + half_d + idx);
            const float4 c_4  = *reinterpret_cast<const float4*>(cos + c_base + idx);
            const float4 s_4  = *reinterpret_cast<const float4*>(sin + s_base + idx);

            float4 o1_4, o2_4;
            o1_4.x = fmaf(x1_4.x, c_4.x, -(x2_4.x * s_4.x));
            o1_4.y = fmaf(x1_4.y, c_4.y, -(x2_4.y * s_4.y));
            o1_4.z = fmaf(x1_4.z, c_4.z, -(x2_4.z * s_4.z));
            o1_4.w = fmaf(x1_4.w, c_4.w, -(x2_4.w * s_4.w));

            o2_4.x = fmaf(x2_4.x, c_4.x, x1_4.x * s_4.x);
            o2_4.y = fmaf(x2_4.y, c_4.y, x1_4.y * s_4.y);
            o2_4.z = fmaf(x2_4.z, c_4.z, x1_4.z * s_4.z);
            o2_4.w = fmaf(x2_4.w, c_4.w, x1_4.w * s_4.w);

            *reinterpret_cast<float4*>(out + o_base + idx) = o1_4;
            *reinterpret_cast<float4*>(out + o_base + half_d + idx) = o2_4;
        }
        return;
    }

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

    // Vectorized 128-bit backward fast path
    if (stride_xb == stride_ob && stride_xd == 1 && stride_cd == 1 && stride_sd == 1 && stride_od == 1 && (half_d % 4 == 0) && std::is_same_v<scalar_t, float>) {
        const int num_vec = half_d / 4;
        const int dx_base = b * stride_xb + h * stride_xh + n * stride_xn;
        const int do_base = b * stride_ob + h * stride_oh + n * stride_on;
        const int c_base  = b * stride_cb + h * stride_ch + n * stride_cn;
        const int s_base  = b * stride_sb + h * stride_sh + n * stride_sn;

        for (int vi = tid; vi < num_vec; vi += blockDim.x) {
            const int idx = vi * 4;
            const float4 do1_4 = *reinterpret_cast<const float4*>(dout + do_base + idx);
            const float4 do2_4 = *reinterpret_cast<const float4*>(dout + do_base + half_d + idx);
            const float4 c_4   = *reinterpret_cast<const float4*>(cos + c_base + idx);
            const float4 s_4   = *reinterpret_cast<const float4*>(sin + s_base + idx);

            float4 dx1_4, dx2_4;
            dx1_4.x = fmaf(do1_4.x, c_4.x, do2_4.x * s_4.x);
            dx1_4.y = fmaf(do1_4.y, c_4.y, do2_4.y * s_4.y);
            dx1_4.z = fmaf(do1_4.z, c_4.z, do2_4.z * s_4.z);
            dx1_4.w = fmaf(do1_4.w, c_4.w, do2_4.w * s_4.w);

            dx2_4.x = fmaf(do2_4.x, c_4.x, -(do1_4.x * s_4.x));
            dx2_4.y = fmaf(do2_4.y, c_4.y, -(do1_4.y * s_4.y));
            dx2_4.z = fmaf(do2_4.z, c_4.z, -(do1_4.z * s_4.z));
            dx2_4.w = fmaf(do2_4.w, c_4.w, -(do1_4.w * s_4.w));

            *reinterpret_cast<float4*>(dx + dx_base + idx) = dx1_4;
            *reinterpret_cast<float4*>(dx + dx_base + half_d + idx) = dx2_4;
        }
        return;
    }

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
