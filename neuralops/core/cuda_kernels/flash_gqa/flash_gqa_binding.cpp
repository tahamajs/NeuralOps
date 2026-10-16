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
    auto out = torch::empty_like(q);
    flash_gqa_fwd_cuda(out, q, k, v, is_causal, static_cast<float>(scale));
    return out;
}
