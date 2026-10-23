// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void liquid_attention_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ delta,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ prev_state,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ final_state,
    int B, int L, int D
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (d >= D || b >= B) return;

    float a_val = fabsf(static_cast<float>(A[d]));
    float h = prev_state ? static_cast<float>(prev_state[b * D + d]) : 0.0f;

    for (int t = 0; t < L; ++t) {
        int idx = (b * L + t) * D + d;
        float dt = fmaxf(static_cast<float>(delta[idx]), 0.0f);
        float decay = expf(-fmaxf(dt * a_val, 0.0f));
        float in_val = static_cast<float>(x[idx]);

        h = decay * h + (1.0f - decay) * in_val;
        h = fminf(fmaxf(h, -65504.0f), 65504.0f);

        float sig = 1.0f / (1.0f + expf(-in_val));
        out[idx] = static_cast<scalar_t>(h * sig);
    }

    if (final_state) {
        final_state[b * D + d] = static_cast<scalar_t>(h);
    }
}

std::vector<torch::Tensor> liquid_attention_fwd(
    const torch::Tensor& x,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const c10::optional<torch::Tensor>& prev_state
) {
    CHECK_INPUT(x);
    CHECK_INPUT(delta);
    CHECK_INPUT(A);

    int B = x.size(0);
    int L = x.size(1);
    int D = x.size(2);

    auto out = torch::empty_like(x);
    auto final_state = torch::empty({B, D}, x.options());

    const void* prev_ptr = nullptr;
    if (prev_state.has_value() && prev_state.value().defined()) {
        CHECK_INPUT(prev_state.value());
        prev_ptr = prev_state.value().data_ptr();
    }

    dim3 block(128);
    dim3 grid((D + 127) / 128, B);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "liquid_attention_fwd", ([&] {
            liquid_attention_fwd_kernel<scalar_t><<<grid, block>>>(
                x.data_ptr<scalar_t>(),
                delta.data_ptr<scalar_t>(),
                A.data_ptr<scalar_t>(),
                prev_ptr ? reinterpret_cast<const scalar_t*>(prev_ptr) : nullptr,
                out.data_ptr<scalar_t>(),
                final_state.data_ptr<scalar_t>(),
                B, L, D
            );
        })
    );

    return {out, final_state};
}
