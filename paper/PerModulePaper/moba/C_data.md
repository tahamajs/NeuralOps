# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS_ABS}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |
| LoRA rank | {{LORA_RANK}} |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS_ABS}} |
| Accuracy | {{R_ACC}}% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS_ABS}} |
| Accuracy | {{G_ACC}}% |

## C.4 Preservation Gate Ablation

| Init | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0.0 | {{D_G0}} | {{R_G0}} | {{G_G0}} |
| 2.0 | {{D_PRES}} | {{R_PRES}} | {{G_PRES}} |
| 4.0 | {{D_G4}} | {{R_G4}} | {{G_G4}} |

## C.5 Decay Ablation

| Decay | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Learned | {{D_LEARNED}} | {{R_LEARNED}} | {{G_LEARNED}} |
| Fixed 0.99 | {{D_FIXED99}} | {{R_FIXED99}} | {{G_FIXED99}} |
| Fixed 0.9 | {{D_FIXED9}} | {{R_FIXED9}} | {{G_FIXED9}} |
| None | {{D_NODECAY}} | {{R_NODECAY}} | {{G_NODECAY}} |

## C.6 Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | {{D_R4}} | {{R_R4}} | {{G_R4}} |
| 8 | {{D_R8}} | {{R_R8}} | {{G_R8}} |
| 16 | {{D_R16}} | {{R_R16}} | {{G_R16}} |
| 32 | {{D_R32}} | {{R_R32}} | {{G_R32}} |
| 64 | {{D_R64}} | {{R_R64}} | {{G_R64}} |

## C.7 Training Loss

### DistilBERT
```
Step  Loss
1     {{D_S1}}
5     {{D_S5}}
...
50    {{D_S50}}
```

## C.8 Scaling

| L | Standard | Mamba | Ratio |
|---|---|---|---|
| 64 | {{T64_STD}} | {{T64_MAMBA}} | {{R64}}× |
| 128 | {{T128_STD}} | {{T128_MAMBA}} | {{R128}}× |
| 256 | {{T256_STD}} | {{T256_MAMBA}} | {{R256}}× |
| 512 | {{T512_STD}} | {{T512_MAMBA}} | {{R512}}× |

## C.9 Reproduction Commands

```bash
python run_all_4_mamba.py --recovery_steps 50
```
```

---