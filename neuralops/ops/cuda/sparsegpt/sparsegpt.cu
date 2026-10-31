// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void sparsegpt_prune_fwd(
    const float* __restrict__ W,
    const float* __restrict__ threshold,
    const float* __restrict__ inv_2h,
    float* __restrict__ out,
    int rows,
    int cols
) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= cols) return;

    int idx = row * cols + col;
    float w = W[idx];
    float inv = (inv_2h != nullptr) ? inv_2h[col] : 1.0f;
    float saliency = (w * w) * inv;
    float thresh = threshold[row];

    out[idx] = (saliency > thresh) ? w : 0.0f;
}

torch::Tensor sparsegpt_prune_forward(
    const torch::Tensor& W,
    double sparsity,
    const c10::optional<torch::Tensor>& diag_h_inv_opt
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

    torch::Tensor diag_h_inv;
    if (diag_h_inv_opt.has_value() && diag_h_inv_opt.value().defined()) {
        diag_h_inv = diag_h_inv_opt.value().to(W.device(), torch::kFloat32).contiguous();
    } else {
        diag_h_inv = torch::full({cols}, 0.5f, W.options());
    }

    auto inv_2h = (1.0f / (2.0f * torch::clamp(diag_h_inv, /*min=*/1e-8f))).contiguous();
    auto saliency = (W * W) * inv_2h.unsqueeze(0);

    int64_t k = std::max(static_cast<int64_t>(cols * sparsity), static_cast<int64_t>(1));
    auto threshold = std::get<0>(torch::kthvalue(saliency, k, /*dim=*/1, /*keepdim=*/false)).contiguous();

    auto out = torch::empty_like(W);

    int threads = 256;
    dim3 block(threads);
    dim3 grid((cols + threads - 1) / threads, rows);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sparsegpt_prune_fwd<<<grid, block, 0, stream>>>(
        W.data_ptr<float>(),
        threshold.data_ptr<float>(),
        inv_2h.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}
