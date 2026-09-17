# 4. Method Formulation
1. **Multi-Scale Window Partition**: Heads are partitioned into 3 groups with window sizes 32, 64, and 128.
2. **Global Integration Branch**: Complementary heads retain linear kernel attention.
3. **LoRA Recovery Layer**: Integrated parameter-efficient recovery tuning.
