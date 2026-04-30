cd /Users/tahamajs/Documents/uni/Project/Codes/framework/compression

# Create directory structure
mkdir -p core svd_based adaptive_svd cur_cr_decomp tt_decomposition kronecker low_rank_sparse quantization pruning hashed moe kv_cache activation_aware advanced kernel_methods polynomial legacy

# Create __init__.py files
for dir in core svd_based adaptive_svd cur_cr_decomp tt_decomposition kronecker low_rank_sparse quantization pruning hashed moe kv_cache activation_aware advanced kernel_methods polynomial legacy; do
    touch $dir/__init__.py
done

# Move files (examples - adjust as needed)
mv svd_linear.py svd_based/ 2>/dev/null
mv svd_compressor.py svd_based/ 2>/dev/null
mv dobi_svd.py svd_based/ 2>/dev/null
mv salr.py svd_based/ 2>/dev/null
mv erc_svd.py svd_based/ 2>/dev/null
mv gfws_d_compressor.py svd_based/ 2>/dev/null
mv mgie_svd.py svd_based/ 2>/dev/null

mv adasvd_compressor.py adaptive_svd/ 2>/dev/null
mv aa_svd_compressor.py adaptive_svd/ 2>/dev/null
mv adaptive_ranks.py adaptive_svd/ 2>/dev/null

mv cur_compressor.py cur_cr_decomp/ 2>/dev/null
mv cur_linear.py cur_cr_decomp/ 2>/dev/null
mv cr_compressor.py cur_cr_decomp/ 2>/dev/null
mv modegpt_compressor.py cur_cr_decomp/ 2>/dev/null

mv tt_linear.py tt_decomposition/ 2>/dev/null
mv tt_compressor.py tt_decomposition/ 2>/dev/null
mv soka_compressor.py tt_decomposition/ 2>/dev/null

mv kronecker_linear.py kronecker/ 2>/dev/null
mv kronecker_compressor.py kronecker/ 2>/dev/null

mv rpca_compressor.py low_rank_sparse/ 2>/dev/null
mv grecodec_compressor.py low_rank_sparse/ 2>/dev/null
mv enlighten_compressor.py low_rank_sparse/ 2>/dev/null
mv frandi_compressor.py low_rank_sparse/ 2>/dev/null
mv slim_compressor.py low_rank_sparse/ 2>/dev/null
mv nmf_compressor.py low_rank_sparse/ 2>/dev/null

mv uniql_compressor.py quantization/ 2>/dev/null
mv mlorq_compressor.py quantization/ 2>/dev/null
mv kmeans_quantize.py quantization/ 2>/dev/null
mv faq_quantizer.py quantization/ 2>/dev/null
mv coopq_quantizer.py quantization/ 2>/dev/null
mv astro_quantizer.py quantization/ 2>/dev/null
mv huffman_coder.py quantization/ 2>/dev/null
mv nli_softmax.py quantization/ 2>/dev/null

mv slicegpt.py pruning/ 2>/dev/null
mv sparsegpt.py pruning/ 2>/dev/null
mv flap.py pruning/ 2>/dev/null
mv oats.py pruning/ 2>/dev/null
mv mamba_shedder.py pruning/ 2>/dev/null
mv armor_pruner.py pruning/ 2>/dev/null
mv ast_pruner.py pruning/ 2>/dev/null
mv admm_pruner.py pruning/ 2>/dev/null
mv admm_compressor.py pruning/ 2>/dev/null
mv bonsai_pruner.py pruning/ 2>/dev/null
mv proxsparse_pruner.py pruning/ 2>/dev/null
mv come.py pruning/ 2>/dev/null
mv replaceme_compressor.py pruning/ 2>/dev/null

mv hashed_linear.py hashed/ 2>/dev/null
mv roast_hashed_linear.py hashed/ 2>/dev/null
mv hashed_net_compressor.py hashed/ 2>/dev/null

mv moe_svd.py moe/ 2>/dev/null
mv cur_moe.py moe/ 2>/dev/null
mv tomoe_compressor.py moe/ 2>/dev/null

mv delta_kv.py kv_cache/ 2>/dev/null
mv hybridkv_compressor.py kv_cache/ 2>/dev/null
mv zerokv_neo_compressor.py kv_cache/ 2>/dev/null
mv cachenotes_compressor.py kv_cache/ 2>/dev/null

mv flat_llm_compressor.py activation_aware/ 2>/dev/null
mv flat_llm.py activation_aware/ 2>/dev/null
mv optimal_subspace_projection.py activation_aware/ 2>/dev/null
mv compot_compressor.py activation_aware/ 2>/dev/null
mv lillama_compressor.py activation_aware/ 2>/dev/null
mv a3_compressor.py activation_aware/ 2>/dev/null

mv advanced_methods.py advanced/ 2>/dev/null
mv methods.py advanced/ 2>/dev/null
mv entropy_budget_allocation.py advanced/ 2>/dev/null
mv sdrm_compressor.py advanced/ 2>/dev/null

mv kernel_attentions.py kernel_methods/ 2>/dev/null
mv rpc_attention.py kernel_methods/ 2>/dev/null
mv galerkin_attention.py kernel_methods/ 2>/dev/null
mv spectraformer_attention.py kernel_methods/ 2>/dev/null
mv tlinformer_attention.py kernel_methods/ 2>/dev/null

mv padre_attention.py polynomial/ 2>/dev/null

# Move legacy/duplicate files
mv misc_compressors.py legacy/ 2>/dev/null
mv pruning_compressors.py legacy/ 2>/dev/null
mv quant_compressors.py legacy/ 2>/dev/null
mv svd_compressors.py legacy/ 2>/dev/null