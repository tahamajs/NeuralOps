import os, csv

modules_str = "fusion_attn,primal,gqa,adaptive_sparse_attention,adaptive_span,adasplash,goldfinch_ahn,halo_regla,nsa_attention,nha,squeezing_heads_distiller,progressive_token_pruning,eat,infini_attention,optimized_hybrid_attention,asvd,rpc,svd_mamba,svd_linear,res_svd,minimal_aurora,minimal_nova,minimal_prism,minimal_svd,svd_llm_v2,mla,dobi,nystrom,unified_rope,hgrn2_gated_deltanet,hgrn2_layer,soafusedattention,adaptive_svd_attention,bam_umoe,block_transformer_hiba,chela_slay,cluster,cosformer,decay_linear_attention,expert_choice_moe,flash3,flash3_fp8,galerkin_attention,gated_delta_net,gated_deltanet_block,griffin_ssd,hymba_transx,jamba_info,vspec,leka_v1,leka_v2,leka_v3,linformer,tpa,lora_attention,xattention,mamba,mixattention_moska,moe_deltakv_paged,nasc_attention,nexus,orion,padre,performer,phantom,projection_factorized_attn,rala,reap_moe,regla,routing,salt,simple_ssm,span_expanded,svd_projection,template_attention,token_compressed,moba,spectra_v2,compressed_mha,svd_compressed,spectron,soa_module,aegis_v5"
mods = [m.strip() for m in modules_str.split(',') if m.strip()]

status_summary = {'ALL_PASS': [], 'PARTIAL': [], 'MISSING': []}
needs_rerun = []

for m in mods:
    model_statuses = {}
    for bb in ['distilbert', 'roberta', 'gpt2']:
        csv_p = f'D:/SVD_Project/results/unified_{m}/{bb}/summary.csv'
        if not os.path.exists(csv_p):
            csv_p = f'D:/SVD_Project/results/{m}/{bb}/summary.csv'
        if not os.path.exists(csv_p):
            model_statuses[bb] = 'MISSING'
        else:
            try:
                with open(csv_p, 'r', encoding='utf-8') as f:
                    r = list(csv.DictReader(f))
                    if len(r) >= 2:
                        st = r[1].get('status', 'UNKNOWN')
                        acc = float(r[1].get('accuracy_pct', 0.0))
                        model_statuses[bb] = f'{st}({acc:.1f}%)'
                    elif len(r) == 1:
                        model_statuses[bb] = 'INCOMPLETE'
                    else:
                        model_statuses[bb] = 'EMPTY'
            except Exception as e:
                model_statuses[bb] = f'ERR:{e}'
    
    vals = list(model_statuses.values())
    if all('PASS' in v and 'PASS_LOW' not in v for v in vals):
        status_summary['ALL_PASS'].append((m, model_statuses))
    elif any(v == 'MISSING' or v == 'INCOMPLETE' or v == 'EMPTY' for v in vals):
        status_summary['MISSING'].append((m, model_statuses))
        needs_rerun.append(m)
    else:
        # has some FAIL or PASS_LOW
        status_summary['PARTIAL'].append((m, model_statuses))
        needs_rerun.append(m)

print(f'Total 83 modules checked: {len(mods)}')
print(f'ALL_PASS count: {len(status_summary["ALL_PASS"])}')
print(f'PARTIAL/NEEDS_WORK count: {len(status_summary["PARTIAL"])}')
print(f'MISSING count: {len(status_summary["MISSING"])}')
print('\n--- PARTIAL/NEEDS_WORK ---')
for m, st in status_summary['PARTIAL']:
    print(f'{m}: {st}')
print('\n--- MISSING ---')
for m, st in status_summary['MISSING']:
    print(f'{m}: {st}')
print(f'\nTotal modules needing run/rerun: {len(needs_rerun)}')
print(','.join(needs_rerun))
