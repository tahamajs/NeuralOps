# 1. Introduction
Previous hybrid architectures often sacrificed physical parameter compactness to accommodate auxiliary adapters and dual-branch routing networks. `aegis_v6` achieves both goals simultaneously: it replaces dense linear layers with true rank-$r$ factorized projections while seamlessly uniting sliding-window local attention with global linear kernel streams.
