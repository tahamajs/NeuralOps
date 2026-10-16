#include <torch/extension.h>
torch::Tensor reformer_fwd(const torch::Tensor& qk, const torch::Tensor& v, const torch::Tensor& hash_buckets);
