// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

void sliding_window_attn_fwd_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int window_size,
    int num_sinks,
    float scale
);

torch::Tensor sliding_window_attn_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    int64_t window_size,
    int64_t num_sinks,
    double scale
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "sliding_window: all input tensors must be on CUDA");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "sliding_window: inputs must be 4D tensors [B, H, N, D]");
    TORCH_CHECK(q.sizes() == k.sizes() && k.sizes() == v.sizes(),
                "sliding_window: q, k, v shapes must match");
    TORCH_CHECK(window_size >= 0 && num_sinks >= 0,
                "sliding_window: window_size and num_sinks must be non-negative");

    auto orig_dtype = q.scalar_type();
    auto q_c = (orig_dtype == torch::kFloat32 ? q : q.to(torch::kFloat32)).contiguous();
    auto k_c = (k.scalar_type() == torch::kFloat32 ? k : k.to(torch::kFloat32)).contiguous();
    auto v_c = (v.scalar_type() == torch::kFloat32 ? v : v.to(torch::kFloat32)).contiguous();

    auto out = torch::zeros_like(q_c);

    sliding_window_attn_fwd_cuda(
        out, q_c, k_c, v_c,
        static_cast<int>(window_size),
        static_cast<int>(num_sinks),
        static_cast<float>(scale)
    );

    if (orig_dtype != torch::kFloat32) {
        return out.to(orig_dtype);
    }
    return out;
}
