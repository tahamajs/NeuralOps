// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void wanda_prune_fwd(
    const float* __restrict__ W,
    const float* __restrict__ threshold,
    const float* __restrict__ act_norm,
    float* __restrict__ out,
    int rows,
    int cols
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;

    int idx = row * cols + col;
    float w = W[idx];
    float a = (act_norm != nullptr) ? act_norm[col] : 1.0f;
    float metric = fabsf(w) * a;
    float thresh = threshold[row];

    out[idx] = (metric > thresh) ? w : 0.0f;
}

torch::Tensor wanda_prune_forward(
    const torch::Tensor& W,
    double sparsity,
    const c10::optional<torch::Tensor>& act_norm_opt
) {
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "W must be float32");
    TORCH_CHECK(W.dim() == 2, "W must be 2D");

    int rows = W.size(0);
    int cols = W.size(1);

    if (sparsity <= 0.0) {
        return W.clone();
    }

    torch::Tensor act_norm;
    bool has_act = act_norm_opt.has_value() && act_norm_opt.value().defined();
    if (has_act) {
        act_norm = act_norm_opt.value().to(W.device(), torch::kFloat32).contiguous();
    } else {
        act_norm = torch::ones({cols}, W.options());
    }

    auto metric = torch::abs(W) * act_norm.unsqueeze(0);
    int64_t k = std::max(static_cast<int64_t>(cols * sparsity), static_cast<int64_t>(1));
    auto threshold = std::get<0>(torch::kthvalue(metric, k, /*dim=*/1, /*keepdim=*/false)).contiguous();

    auto out = torch::empty_like(W);

    int threads = 256;
    dim3 block(threads);
    dim3 grid((cols + threads - 1) / threads, rows);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    wanda_prune_fwd<<<grid, block, 0, stream>>>(
        W.data_ptr<float>(),
        threshold.data_ptr<float>(),
        act_norm.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}
