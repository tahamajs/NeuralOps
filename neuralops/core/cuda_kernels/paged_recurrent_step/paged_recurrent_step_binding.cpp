#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor paged_recurrent_step_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, torch::Tensor& page_pool, const torch::Tensor& block_tables, float decay);
