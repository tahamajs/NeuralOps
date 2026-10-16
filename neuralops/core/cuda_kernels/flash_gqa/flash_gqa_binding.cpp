#include <torch/extension.h>

void flash_gqa_fwd_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool is_causal,
    float scale
);

torch::Tensor flash_gqa_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    double scale
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "flash_gqa: all input tensors must be on CUDA");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "flash_gqa: inputs must be 4D tensors [B, H, N, D]");
    TORCH_CHECK(k.sizes() == v.sizes(),
                "flash_gqa: k and v shapes must match");
    TORCH_CHECK(q.size(0) == k.size(0),
                "flash_gqa: batch size mismatch between q and kv");
    TORCH_CHECK(q.size(3) == k.size(3),
                "flash_gqa: head_dim mismatch between q and kv");
    TORCH_CHECK(q.size(1) % k.size(1) == 0,
                "flash_gqa: num_heads (q) must be divisible by num_kv_heads (k)");

    const int D = static_cast<int>(q.size(3));
    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
                "flash_gqa: only head dimensions 32, 64, 128, and 256 are supported by CUDA kernel");

    auto q_c = q.contiguous();
    auto k_c = k.to(q.scalar_type()).contiguous();
    auto v_c = v.to(q.scalar_type()).contiguous();

    auto out = torch::empty_like(q_c);

    flash_gqa_fwd_cuda(out, q_c, k_c, v_c, is_causal, static_cast<float>(scale));

    return out;
}
