# Project SOA – Shared‑Basis BERT for Core ML / ANE

## Build ONNX
```bash
python -m soa.export_onnx --rank 64 --onnx_path soa_bert.onnx

## Convert to Core ML 6 (mlprogram, ANE)
bash
python -m soa.export_coreml --onnx_path soa_bert.onnx --mlmodel_path soa_bert.mlmodel

## Rank Sweep
bash
python -m soa.rank_sweep


---

# ✅ Notes / Compatibility Guarantees

- **Fused QKV**: implemented in `SOAFusedAttention`  
- **Shared global basis**: `self.basis` in model, frozen  
- **BERT‑base dimensions**: 12 layers, 768 hidden, 3072 FFN  
- **ONNX opset 17**: enforced in export  
- **Core ML 6**: `mlprogram`, `CPU_AND_NE`  
- **ANE‑friendly ops only**: MatMul, BatchMatMul, Softmax, LN, GELU

---

If you want me to add **fine‑tuning scripts**, **SVD‑based initialization into the full model weights**, or **GLUE eval**, say the word and I’ll append those next.