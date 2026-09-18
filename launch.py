import subprocess
import os

with open("logs/remaining_sweep.log", "w") as f_out, open("logs/remaining_sweep_err.log", "w") as f_err:
    subprocess.Popen([
        "python", "-u", "Codes/framework/run_quality_sweep.py",
        "--model", "Qwen/Qwen2.5-0.5B",
        "--dataset", "qb_eval",
        "--modules", "mla_attention,moe_deltakv_paged,nasc_attention,nexus,nha_attention,nli_softmax,nsa_attention,nystrom_attention,optimized_hybrid_attention,orion_attention,padre_attention,performer_attention,primal_attention,projection_factorized_attn,qcqa_attention,rala_linear_attention,reap_moe,replace_attention,res_svd_attention,routing_attention,rpc_attention,salt,samba_m3,samba_moe,simple_ssm,smart_weighted_ensemble,soa_fused_attention,span_expanded_attention,spectraformer_attention,spectron_attn,squeezing_heads_module,svd_compressed_module,svd_linear,svd_llm_v2_attention,svd_mamba,svd_sequence_attention,tlinformer_attention,token_compressed_module,tp_attention,upi_mamba,vig_attention,wgqa_attention,zeros_attention"
    ], stdout=f_out, stderr=f_err, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
