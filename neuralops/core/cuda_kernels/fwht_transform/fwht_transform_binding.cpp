#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor fwht_fwd(const torch::Tensor& x, int B_dim);
