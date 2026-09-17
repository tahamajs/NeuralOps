# 1. Introduction
Modern transformers require both fine-grained local context capture and long-range sequence representations. `aegis_v1` decouples these duties across attention heads: a subset of heads executes local sliding-window attention, while complementary heads compute global linear attention using kernel approximations.
