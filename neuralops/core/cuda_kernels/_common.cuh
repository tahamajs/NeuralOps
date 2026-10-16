#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_KERNEL_LAUNCH_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err)); \
    } while (0)

namespace neuralops {
namespace cuda {

template <typename T>
__device__ __forceinline__ float to_float(T val);

template <>
__device__ __forceinline__ float to_float<float>(float val) { return val; }

template <>
__device__ __forceinline__ float to_float<double>(double val) { return static_cast<float>(val); }

template <>
__device__ __forceinline__ float to_float<at::Half>(at::Half val) {
    return __half2float(*reinterpret_cast<const __half*>(&val));
}

template <>
__device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}

template <>
__device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 val) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&val));
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template <typename T>
__device__ __forceinline__ T from_float(float val);

template <>
__device__ __forceinline__ float from_float<float>(float val) { return val; }

template <>
__device__ __forceinline__ double from_float<double>(float val) { return static_cast<double>(val); }

template <>
__device__ __forceinline__ at::Half from_float<at::Half>(float val) {
    __half h = __float2half(val);
    return *reinterpret_cast<at::Half*>(&h);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float val) {
    return __float2half(val);
}

template <>
__device__ __forceinline__ at::BFloat16 from_float<at::BFloat16>(float val) {
    __nv_bfloat16 bf = __float2bfloat16(val);
    return *reinterpret_cast<at::BFloat16*>(&bf);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float s_warp_sum[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        s_warp_sum[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp_sum[threadIdx.x] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float s_warp_max[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warp_reduce_max(val);
    if (lane == 0) {
        s_warp_max[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp_max[threadIdx.x] : -1e30f;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }
    return val;
}

} // namespace cuda
} // namespace neuralops
