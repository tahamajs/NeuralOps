#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor logcumsumexp_fwd(const torch::Tensor& x, int dim);


