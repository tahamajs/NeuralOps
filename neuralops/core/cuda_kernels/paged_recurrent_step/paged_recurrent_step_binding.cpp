#include <torch/extension.h>

torch::Tensor paged_recurrent_step_fwd(
    torch::Tensor& page_pool,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& block_tables,
    int page_slot,
    float decay
);


