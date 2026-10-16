#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor entropy_row_reduction_fwd(const torch::Tensor& scores);
