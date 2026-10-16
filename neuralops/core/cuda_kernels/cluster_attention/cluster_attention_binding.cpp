#include <torch/extension.h>
torch::Tensor cluster_attn_fwd(const torch::Tensor& q, const torch::Tensor& cluster_centroids, const torch::Tensor& v, const torch::Tensor& cluster_indices);
