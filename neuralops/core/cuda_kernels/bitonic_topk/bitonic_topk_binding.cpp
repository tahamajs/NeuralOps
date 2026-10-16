#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor bitonic_topk_indices_fwd(const torch::Tensor& x, int K);
