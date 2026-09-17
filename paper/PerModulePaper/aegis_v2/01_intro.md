# 1. Introduction
While early hybrid attention models often inflated total parameters due to auxiliary adapter layers, `aegis_v2` enforces true physical parameter compression by factorizing the base linear projection weights into narrow rank-$r$ bottleneck matrices ($r=128$) while retaining local sliding-window precision.
