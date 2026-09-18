@echo off
cd /d D:\SVD_Project

echo === Baseline ===
python -u Codes/framework/train_lra.py --task listops --module none --output_dir results/lra_listops_none/
python -u Codes/framework/train_lra.py --task imdb --module none --output_dir results/lra_imdb_none/

for %%M in (cosformer performer_attention samba_m3 mamba hla_inject gated_deltanet_block galerkin phantom jamba unified_rope) do (
    echo === Running ListOps with %%M ===
    python -u Codes/framework/train_lra.py --task listops --module %%M --output_dir results/lra_listops_%%M/

    echo === Running IMDb with %%M ===
    python -u Codes/framework/train_lra.py --task imdb --module %%M --output_dir results/lra_imdb_%%M/
)
