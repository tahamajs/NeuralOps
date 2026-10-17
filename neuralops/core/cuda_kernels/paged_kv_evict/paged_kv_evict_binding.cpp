#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> paged_kv_compact_fwd(
    const torch::Tensor& k_in,
    const torch::Tensor& v_in,
    const torch::Tensor& keep_indices
);


