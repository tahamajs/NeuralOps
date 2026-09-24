import sys
sys.path.insert(0, 'Codes')
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig

MODULES = [
    ('aegis_hybrid', 'inject_aegis_hybrid'),
    ('aegis_hybrid_attention', 'inject_aegis_hybrid_attention'),
    ('aegis_spectron_fp8', 'inject_aegis_spectron_fp8'),
    ('aegis_v1', 'inject_aegis_v1'),
    ('aegis_v2', 'inject_aegis_v2'),
    ('aegis_v3', 'inject_aegis_v3'),
    ('aegis_v4', 'inject_aegis_v4'),
    ('aegis_v5', 'inject_aegis_v5'),
    ('aegis_v6', 'inject_aegis_v6'),
    ('spectron', 'inject_spectron_layer'),
    ('spectron_attention', 'inject_spectron_attention'),
]

MODELS = [
    ('distilbert', 'distilbert-base-uncased'),
    ('roberta', 'roberta-base'),
    ('gpt2', 'gpt2'),
]

fails = 0
total = 0
for mname, mpath in MODELS:
    cfg = AutoConfig.from_pretrained(mpath, local_files_only=True)
    if mname == 'gpt2' and getattr(cfg, 'pad_token_id', None) is None:
        cfg.pad_token_id = cfg.eos_token_id

    for mod_name, inject_name in MODULES:
        total += 1
        try:
            model = AutoModelForSequenceClassification.from_pretrained(mpath, config=cfg, local_files_only=True)
            if mname == 'gpt2' and getattr(model.config, 'pad_token_id', None) is None:
                model.config.pad_token_id = model.config.eos_token_id
            model.eval()
            mod = __import__(f'framework.attentions.{mod_name}', fromlist=[inject_name])
            inject = getattr(mod, inject_name)
            
            ids = torch.randint(0, 1000, (2, 16))
            attn = torch.ones_like(ids)
            with torch.no_grad():
                before = model(ids, attention_mask=attn).logits.clone()
            
            res = inject(model=model, model_config=cfg)
            with torch.no_grad():
                after = model(ids, attention_mask=attn).logits
            diff = (before - after).abs().max().item()
            print(f"[{mname}] {mod_name}: PASS ({diff:.4f})", flush=True)
        except Exception as e:
            fails += 1
            print(f"[{mname}] {mod_name}: FAIL ({type(e).__name__}: {str(e)[:60]})", flush=True)

print(f"\nFINAL SUMMARY: {total - fails}/{total} passed", flush=True)
