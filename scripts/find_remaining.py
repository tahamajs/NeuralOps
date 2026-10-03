import os
import sys

distil_queue = ['svd_llm','kmeans','slim','tomoe','sparsegpt','aatc_xkv','coopq','cur_moe','delta_kv','drone','dynamic_entropy','dynamic_rank','enlighten','flap','flat_llm','fwsvd','gfws_d','hashednets','hybridkv','kvtc','liftr','lillama','mamba_shedder','pca_quant','proxsparse','roast','soka','spectral_residual_lora','zerokv_neo']
roberta_queue = ['a3_compressor','aa_svd_compressor','aasvd','adaptive_compressor','adaptive_ranks','adasvd_compressor','admm_compressor','come','compot_compressor','cr_compressor','cur_compressor','cur_linear','cure','direct_svd','dobi_svd','erc_svd','frandi','frandi_compressor','grecodec','grecodec_compressor','hybrid_svd_layer','kronecker_compressor','lsc_compressor','mgie_svd','mlorq_compressor','modegpt','modegpt_compressor','nmf_compressor','oats','oats_compressor','rotational_slicegpt_int4','rpca','rpca_compressor','saes_svd','saes_svd_compressor','saes_svd_v2','salr','slicegpt','slicegpt_compressor','slicegpt_pca','subspace_projection','svd_linear','swift_svd_act','swift_svd_compressor','uniql_compressor','svd_llm','kmeans','slim','tomoe']

base_d = r'D:\SVD_Project\results\04_rank_sweeps\distilbert\deep_rank_distilbert'
base_r = r'D:\SVD_Project\results\04_rank_sweeps\roberta\deep_rank_roberta'

d_done = [m for m in distil_queue if os.path.exists(os.path.join(base_d, m, 'rank_curve.csv'))]
r_done = [m for m in roberta_queue if os.path.exists(os.path.join(base_r, m, 'rank_curve.csv'))]

d_rem = [m for m in distil_queue if m not in d_done]
r_rem = [m for m in roberta_queue if m not in r_done]

print(f"DISTILBERT_DONE={len(d_done)}/{len(distil_queue)}")
print(f"DISTILBERT_REMAINING={len(d_rem)}: {d_rem}")
print(f"ROBERTA_DONE={len(r_done)}/{len(roberta_queue)}")
print(f"ROBERTA_REMAINING={len(r_rem)}: {r_rem}")
