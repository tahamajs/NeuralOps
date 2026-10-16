#include <torch/extension.h>

void paged_attention_v1_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    int block_size,
    int max_context_len,
    float scale
);

torch::Tensor paged_attention_v1(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int64_t block_size,
    int64_t max_context_len,
    double scale
) {
    auto out = torch::zeros_like(q);
    paged_attention_v1_cuda(
        out,
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        static_cast<int>(block_size),
        static_cast<int>(max_context_len),
        static_cast<float>(scale)
    );
    return out;
}
