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
#include "_common.cuh"

torch::Tensor spectral_entropy_svd_gate_fwd(
    const torch::Tensor& x,
    const torch::Tensor& v_r,
    const torch::Tensor& u_r,
    const int effective_rank,
    const c10::optional<torch::Tensor>& s_weights,
    const c10::optional<torch::Tensor>& bias
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v_r.is_cuda(), "v_r must be a CUDA tensor");
    TORCH_CHECK(u_r.is_cuda(), "u_r must be a CUDA tensor");

    auto orig_shape = x.sizes();
    auto x_2d = x.reshape({-1, orig_shape.back()});
    const int64_t K = x_2d.size(1);

    auto V = v_r;
    if (V.size(0) == K) {
        V = V.t();
    }
    const int64_t R_MAX = V.size(0);
    const int64_t eff_r = std::min((int64_t)effective_rank, R_MAX);

    auto U = u_r;
    if (U.size(0) != R_MAX && U.size(1) == R_MAX) {
        U = U.t();
    }
    const int64_t N = U.size(1);

    auto V_sub = V.slice(0, 0, eff_r);
    auto U_sub = U.slice(0, 0, eff_r);

    auto orig_dtype = x.scalar_type();
    auto x_f32 = (orig_dtype != torch::kFloat32) ? x_2d.to(torch::kFloat32) : x_2d;
    auto V_f32 = (orig_dtype != torch::kFloat32) ? V_sub.to(torch::kFloat32) : V_sub;
    auto U_f32 = (orig_dtype != torch::kFloat32) ? U_sub.to(torch::kFloat32) : U_sub;

    auto z = at::matmul(x_f32, V_f32.t());

    if (s_weights.has_value() && s_weights.value().defined()) {
        auto sw = s_weights.value().slice(0, 0, eff_r).to(torch::kFloat32).view({1, -1});
        z = z * sw;
    }

    auto y = at::matmul(z, U_f32);

    if (bias.has_value() && bias.value().defined()) {
        y = y + bias.value().to(torch::kFloat32).view({1, -1});
    }

    if (orig_dtype != torch::kFloat32) {
        y = y.to(orig_dtype);
    }


    std::vector<int64_t> final_shape;
    for (size_t i = 0; i < orig_shape.size() - 1; ++i) {
        final_shape.push_back(orig_shape[i]);
    }
    final_shape.push_back(N);
    return y.view(final_shape);
}

