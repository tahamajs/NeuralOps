#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor entmax15_fwd(const torch::Tensor& scores, int n_iter);
torch::Tensor entmax15_bwd(const torch::Tensor& dout, const torch::Tensor& p);
